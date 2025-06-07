#!/usr/bin/env python3
"""
pdf_to_markdown_seq.py  – Pipeline con OCR de Gemini y formato GPT
Guarda el resultado en <output-dir>/<nombre_base>.md
"""
import argparse, shutil, base64, time, os
from pathlib import Path
from typing import List
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
import google.generativeai as genai # Import para Gemini

# ---------- reglas de formateo (para OpenAI GPT) ---------- #
RULES = """
Sigue estas reglas **exactas** (¡no improvises nada!):

1. **No crear tablas** bajo ningún concepto a menos que el PDF original tenga una tabla clara.

2. **No inventar** letras (A., B., C.) ni renumerar secciones: respeta la numeración tal cual aparece en el texto.

3. Para **títulos principales** (cabeceras) utiliza `# `:
   - Cualquier línea que esté **completamente** en MAYÚSCULAS —incluso si contiene símbolos como `–`, `/`, `(`, `)` o números— debe marcarse con un solo `# ` al inicio.
   - Ejemplo: `# UNSM – T (CPU) / BIOLOGÍA`

4. Para subtítulos con numeración:
   - `## I. II. III.` para numerales romanos.
   - `### 1. 2. 3.` para numerales arábigos.
   - `#### a. b. c.` para subpuntos.

5. Para viñetas, usa siempre `* ` (asterisco + espacio). No uses `|`.

6. **Negrita** solo en términos exactos que estén entre asteriscos en el PDF.

7. **NO** cursiva, **NO** enlaces, **NO** emojis.

8. **NO** separadores `---` salvo que realmente aparezcan en el PDF.

9. Respeta ortografía, puntuación y el flujo original del texto.

Devuélveme únicamente el bloque Markdown listo para pegar.
"""

# Configuración de clientes
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("La variable de entorno OPENAI_API_KEY no está configurada.")
if not GOOGLE_API_KEY:
    raise ValueError("La variable de entorno GOOGLE_API_KEY no está configurada.")

client_openai = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

# Modelo Gemini para OCR (puedes cambiarlo por uno más reciente si es necesario, ej. 'gemini-1.5-flash')
GEMINI_OCR_MODEL = "gemini-pro-vision" 
# Modelo OpenAI para formateo
OPENAI_FORMAT_MODEL = "gpt-4o-mini"

# ---------- helpers ---------- #
def z(n:int, total:int)->str:
    return str(n).zfill(len(str(total)))

# b64 ya no es necesaria directamente aquí si Gemini maneja PIL Images o bytes directamente.
# La mantenemos por si acaso o para referencia futura.
def b64_encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode()

# Nueva función para OCR con Google Gemini
def ocr_gemini(image_path: Path, max_retries: int = 3, backoff: float = 2.0) -> str:
    """
    Extrae texto de una imagen utilizando Google Gemini Vision.
    Incluye reintentos en caso de fallo.
    """
    img = Image.open(image_path) # Carga la imagen con PIL

    # Prepara el prompt para Gemini
    prompt_parts = [
        "Extrae todo el texto visible en esta imagen. Devuelve solo el texto extraído, sin comentarios adicionales.",
        img, # Pasa el objeto PIL.Image directamente
    ]

    generation_config = genai.types.GenerationConfig(
        # Ajusta estos parámetros según sea necesario. Para OCR, queremos la respuesta más directa.
        # temperature=0.1, # Podría ser útil para reducir alucinaciones si ocurren
        # top_p=0.9,
        # top_k=1, # Para una respuesta más determinista
        max_output_tokens=8192 # Asegúrate de que sea suficiente para el texto de una página
    )
    
    safety_settings = [ # Ajusta según sea necesario, para OCR es improbable que se bloquee contenido
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    model = genai.GenerativeModel(
        model_name=GEMINI_OCR_MODEL,
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt_parts)
            # Verifica si hay texto en la respuesta y si parts no está vacío
            if response.parts:
                return response.text # .text es un helper para concatenar las partes de texto
            elif response.candidates and response.candidates[0].content.parts:
                 # Intenta acceder al texto a través de candidates si .text no está disponible directamente
                full_text = ""
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                         full_text += part.text
                if full_text:
                    return full_text
                else: # Si no hay texto en parts o candidates
                    print(f"Advertencia: Gemini no devolvió texto para {image_path.name} (intento {attempt}). Respuesta: {response}")
                    return "" # Devolver cadena vacía si no hay texto
            else: # Si no hay texto en parts
                print(f"Advertencia: Gemini no devolvió texto para {image_path.name} (intento {attempt}). Respuesta: {response}")
                return "" # Devolver cadena vacía si no hay texto
        except Exception as e:
            if attempt < max_retries:
                wait = backoff * attempt
                print(f"Error en ocr_gemini con {image_path.name} (intento {attempt}/{max_retries}): {e}. Reintentando en {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"Error final en ocr_gemini con {image_path.name} (intento {attempt}/{max_retries}): {e}. Abortando para esta imagen.")
                # Considera si quieres que lance la excepción o devuelva una cadena vacía
                # Por ahora, devolvemos cadena vacía para que el proceso continúe con otras imágenes.
                return "" 
    return "" # En caso de que todos los reintentos fallen sin lanzar excepción

# ---------- pipeline ---------- #
def split_pdf(pdf:Path, out:Path)->List[Path]:
    r=PdfReader(str(pdf)); t=len(r.pages); lst=[]
    for i in tqdm(range(t), desc="Dividir PDF"):
        w=PdfWriter(); w.add_page(r.pages[i])
        p=out/f"{z(i+1,t)}.pdf"
        with p.open("wb") as f: w.write(f)
        lst.append(p)
    return lst

def pdfs_to_jpgs(pdfs:List[Path], out:Path, dpi:int)->List[Path]:
    imgs=[]
    for p in tqdm(pdfs,desc="PDF→JPG"):
        pil_images = convert_from_path(str(p), dpi=dpi, fmt="jpeg")
        if pil_images: # Asegurarse de que convert_from_path devolvió algo
            j=out/p.with_suffix(".jpg").name
            pil_images[0].save(j,"JPEG", quality=95) # Guardar la primera imagen (y única)
            imgs.append(j)
        else:
            print(f"Advertencia: No se pudo convertir {p.name} a JPG.")
    return imgs

# --- Función process utilizando Gemini OCR y OpenAI para formateo ---
def process(jpgs: List[Path], md: Path):
    # 1) Extraer texto con Gemini OCR
    textos_extraidos = []
    print(f"\nIniciando OCR con Google Gemini para {len(jpgs)} imágenes...")
    for p_jpg in tqdm(jpgs, desc="OCR con Gemini"):
        try:
            texto_pagina = ocr_gemini(p_jpg)
            if texto_pagina: # Solo añadir si se extrajo algo
                textos_extraidos.append(texto_pagina)
            else:
                print(f"No se extrajo texto de {p_jpg.name} o la respuesta estaba vacía.")
        except Exception as e: # Captura errores más generales durante la iteración
            print(f"Error procesando {p_jpg.name} con Gemini OCR: {e}. Página omitida.")

    if not textos_extraidos:
        print("No se pudo extraer texto de ninguna página usando Gemini.")
        return

    raw_block_gemini = "\n\n---\n\n".join(textos_extraidos) # Unir textos de cada página con un separador distintivo

    # 2) Formatear TODO el bloque de texto con OpenAI GPT
    print(f"\nFormateando {len(textos_extraidos)} bloques de texto con OpenAI ({OPENAI_FORMAT_MODEL})...")
    mensaje_para_openai = RULES + "\n\nEl siguiente texto fue extraído de varias páginas de un PDF. Por favor, formatea cada sección manteniendo la estructura y aplicando las reglas:\n\n" + raw_block_gemini
    
    try:
        resp_openai = client_openai.chat.completions.create(
            model=OPENAI_FORMAT_MODEL,
            messages=[{"role": "user", "content": mensaje_para_openai}]
        )
        md_block_formateado = resp_openai.choices[0].message.content.strip()

        # 3) Guardar en el .md
        # Leemos el contenido existente solo una vez si el archivo ya tiene algo
        # Para la primera escritura, o si se quiere sobreescribir, el manejo actual está bien.
        # Si se quiere añadir incrementalmente en múltiples llamadas a `process`, se necesitaría leer `md` antes del bucle.
        # Por ahora, el script está diseñado para procesar un PDF completo a la vez.
        
        # md.write_text se llama al final en main() después de que process() completa todo el PDF
        # Así que aquí solo devolvemos el bloque completo.
        # existing_content = md.read_text(encoding="utf-8") if md.exists() and md.stat().st_size > 0 else ""
        # md.write_text(existing_content + md_block_formateado + "\n\n", encoding="utf-8")
        
        # La función process ahora devuelve el bloque de markdown para que main lo escriba
        print(f"✅ Bloque de Markdown formateado por OpenAI listo.")
        return md_block_formateado

    except Exception as e:
        print(f"Error durante el formateo con OpenAI: {e}")
        return None # O manejar el error de otra forma

def main():
    ap = argparse.ArgumentParser(description="Convierte un PDF en Markdown usando Gemini OCR y OpenAI GPT para formateo.")
    ap.add_argument("--input-dir", default="Input_files",
                    help="Directorio de entrada con PDF (por defecto %(default)s)")
    ap.add_argument("--output-dir", default="Output_files",
                    help="Directorio de salida para el Markdown generado (por defecto %(default)s)")
    ap.add_argument("--pdf", required=True,
                    help="Nombre del archivo PDF a procesar dentro de --input-dir")
    ap.add_argument("--dpi", type=int, default=300,
                    help="Resolución DPI para la conversión a JPG (por defecto %(default)s)")
    args = ap.parse_args()

    pdf_path = Path(args.input_dir) / args.pdf
    if not pdf_path.is_file():
        print(f"Error: El archivo PDF {pdf_path} no existe.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Usar directorios temporales dentro del out_dir para páginas e imágenes podría ser más limpio
    # o seguir usando los directorios base como antes. Por ahora, se mantiene igual.
    base_workspace = Path.cwd() 
    pages_dir = base_workspace / "pages_temp" # Cambiado para evitar conflictos si existen "pages"
    images_dir = base_workspace / "images_temp" # Cambiado para evitar conflictos si existen "images"
    pages_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    nombre_base = pdf_path.stem
    md_file_path = out_dir / f"{nombre_base}.md"
    
    # Limpiar el archivo Markdown de salida al inicio de cada ejecución completa
    md_file_path.write_text("", encoding="utf-8") 

    print(f"Procesando {pdf_path.name}...")
    paginas_pdf_individuales = split_pdf(pdf_path, pages_dir)
    if not paginas_pdf_individuales:
        print("No se pudieron dividir las páginas del PDF.")
        return
        
    imagenes_jpg = pdfs_to_jpgs(paginas_pdf_individuales, images_dir, args.dpi)
    if not imagenes_jpg:
        print("No se pudieron convertir las páginas PDF a JPG.")
        return

    # Procesa todas las imágenes y obtiene el bloque de markdown completo
    markdown_completo = process(imagenes_jpg, md_file_path) # md_file_path se pasa pero 'process' ya no escribe directamente

    if markdown_completo:
        md_file_path.write_text(markdown_completo + "\n", encoding="utf-8")
        print(f"\n🎉 Fin. Markdown generado: {md_file_path}")
    else:
        print(f"\n❌ No se pudo generar el archivo Markdown.")

    # Limpieza opcional de directorios temporales
    try:
        shutil.rmtree(pages_dir)
        shutil.rmtree(images_dir)
        print("Directorios temporales limpiados.")
    except Exception as e:
        print(f"Error limpiando directorios temporales: {e}")

if __name__=="__main__":
    main()

