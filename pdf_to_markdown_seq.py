#!/usr/bin/env python3
"""
pdf_to_markdown_seq.py  – Pipeline con OCR de Google Gemini
Guarda el resultado en <output-dir>/<nombre_base>.md
"""
import argparse, shutil, base64, time, os
from pathlib import Path
from typing import List
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
import google.generativeai as genai # Import para Gemini

# Ya no se necesitan RULES para OpenAI

# Configuración de claves API para Google Gemini
# Intentar cargar desde variables de entorno primero, luego desde config_local.py
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    try:
        import config_local
        if hasattr(config_local, 'GOOGLE_API_KEY'):
            GOOGLE_API_KEY = config_local.GOOGLE_API_KEY
            if GOOGLE_API_KEY:
                print("INFO: GOOGLE_API_KEY cargada desde config_local.py")
            else:
                print("Advertencia: GOOGLE_API_KEY está vacía en config_local.py.")
        else:
            # Si config_local existe pero no tiene GOOGLE_API_KEY
            print("Advertencia: config_local.py encontrado, pero GOOGLE_API_KEY no está definida dentro.")

    except ImportError:
        print("Advertencia: config_local.py no encontrado. GOOGLE_API_KEY debe estar como variable de entorno o en config_local.py.")
    except AttributeError:
        print("Advertencia: Atributo GOOGLE_API_KEY no encontrado en config_local.py.")

if not GOOGLE_API_KEY:
    raise ValueError("La GOOGLE_API_KEY no está configurada como variable de entorno ni encontrada/definida en config_local.py.")

genai.configure(api_key=GOOGLE_API_KEY)

# Modelo Gemini para OCR
GEMINI_OCR_MODEL = "gemini-pro-vision" # o "gemini-1.5-flash-latest" si prefieres y está disponible

# ---------- helpers ---------- #
def z(n:int, total:int)->str:
    return str(n).zfill(len(str(total)))

# b64_encode_image se mantiene por si se necesita para otros usos o modelos Gemini que requieran base64 explícito
def b64_encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode()

# Función para OCR con Google Gemini
def ocr_gemini(image_path: Path, max_retries: int = 3, backoff: float = 2.0) -> str:
    """
    Extrae texto de una imagen utilizando Google Gemini Vision.
    Incluye reintentos en caso de fallo.
    """
    img = Image.open(image_path)
    prompt_parts = [
        "Extrae todo el texto visible en esta imagen. Devuelve solo el texto extraído, sin comentarios adicionales, manteniendo los saltos de línea originales si es posible.",
        img,
    ]
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=8192
    )
    safety_settings = [
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
            if response.parts:
                return response.text
            elif response.candidates and response.candidates[0].content.parts:
                full_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text') and part.text)
                if full_text:
                    return full_text
            print(f"Advertencia: Gemini no devolvió texto para {image_path.name} (intento {attempt}). Respuesta: {response}")
            return ""
        except Exception as e:
            if attempt < max_retries:
                wait = backoff * attempt
                print(f"Error en ocr_gemini con {image_path.name} (intento {attempt}/{max_retries}): {e}. Reintentando en {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"Error final en ocr_gemini con {image_path.name} (intento {attempt}/{max_retries}): {e}. Abortando para esta imagen.")
                return ""
    return ""

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
        if pil_images:
            j=out/p.with_suffix(".jpg").name
            pil_images[0].save(j,"JPEG", quality=95)
            imgs.append(j)
        else:
            print(f"Advertencia: No se pudo convertir {p.name} a JPG.")
    return imgs

# --- Función process utilizando SOLO Gemini OCR ---
def process(jpgs: List[Path]) -> str: # Ya no necesita md_path
    textos_extraidos = []
    print(f"\nIniciando OCR con Google Gemini para {len(jpgs)} imágenes...")
    for p_jpg in tqdm(jpgs, desc="OCR con Gemini"):
        try:
            texto_pagina = ocr_gemini(p_jpg)
            if texto_pagina:
                textos_extraidos.append(texto_pagina)
            else:
                print(f"No se extrajo texto de {p_jpg.name} o la respuesta estaba vacía.")
        except Exception as e:
            print(f"Error procesando {p_jpg.name} con Gemini OCR: {e}. Página omitida.")

    if not textos_extraidos:
        print("No se pudo extraer texto de ninguna página usando Gemini.")
        return "" # Devuelve cadena vacía si no hay nada

    # Simplemente une los textos de cada página. El usuario puede querer formatear esto después.
    # Usamos dos saltos de línea para separar el texto de cada página.
    final_markdown_block = "\n\n".join(textos_extraidos)
    print(f"✅ Texto extraído por Gemini listo.")
    return final_markdown_block

def main():
    ap = argparse.ArgumentParser(description="Convierte un PDF en Markdown usando SOLO Google Gemini OCR.")
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

    base_workspace = Path.cwd()
    pages_dir = base_workspace / "pages_temp"
    images_dir = base_workspace / "images_temp"
    pages_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    nombre_base = pdf_path.stem
    md_file_path = out_dir / f"{nombre_base}.md"
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

    markdown_completo = process(imagenes_jpg)

    if markdown_completo:
        md_file_path.write_text(markdown_completo + "\n", encoding="utf-8")
        print(f"\n🎉 Fin. Markdown generado: {md_file_path}")
    else:
        print(f"\n❌ No se pudo generar el archivo Markdown (no se extrajo texto o ocurrió un error).")

    try:
        shutil.rmtree(pages_dir)
        shutil.rmtree(images_dir)
        print("Directorios temporales limpiados.")
    except Exception as e:
        print(f"Error limpiando directorios temporales: {e}")

if __name__=="__main__":
    main()

