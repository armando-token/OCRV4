#!/usr/bin/env python3
"""
pdf_to_markdown_seq.py  – Pipeline 3-en-3 con GPT-4 Vision
Guarda el resultado en <output-dir>/<nombre_base>.md
"""
import argparse, shutil, base64, time
from pathlib import Path
from typing import List
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
from openai import OpenAI            # pip install openai>=1.3.5

# ---------- reglas de formateo ---------- #
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

CHUNK = 3
client = OpenAI()   # necesita OPENAI_API_KEY

# ---------- helper ---------- #
def z(n:int, total:int)->str:
    return str(n).zfill(len(str(total)))

def b64(path:Path)->str:
    return base64.b64encode(path.read_bytes()).decode()

def format_text(text: str) -> str:
    """
    Aplica las reglas definidas en RULES al texto proporcionado mediante GPT.
    Devuelve el bloque Markdown formateado listo para pegar.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": RULES + "\n\n" + text}]
    )
    return resp.choices[0].message.content.strip()

def gpt_vision(imgs: List[Path], max_retries: int = 3, backoff: float = 1.5) -> str:
    """
    Envía imágenes a GPT Vision con RULES y reintentos en caso de fallo.
    """
    msgs = [{"type": "text", "text": RULES}]
    for p in imgs:
        msgs.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(p)}"}})
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": msgs}]
            )
            raw = resp.choices[0].message.content.strip()
            return format_text(raw)
        except Exception as e:
            if attempt < max_retries:
                wait = backoff * attempt
                print(f"Error en gpt_vision (intento {attempt}/{max_retries}): {e}. Reintentando en {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"Error en gpt_vision (intento final {attempt}/{max_retries}): {e}. Abortando.")
                raise

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
        pil=convert_from_path(str(p), dpi=dpi, fmt="jpeg")[0]
        j=out/p.with_suffix(".jpg").name
        pil.save(j,"JPEG"); imgs.append(j)
    return imgs

# --- Simpler process function ---
def process(jpgs: List[Path], md: Path):
    for i in range(0, len(jpgs), CHUNK):
        chunk = jpgs[i:i + CHUNK]
        print(f"🟦 Páginas {chunk[0].stem}-{chunk[-1].stem}")
        md_block = gpt_vision(chunk)
        existing = md.read_text(encoding="utf-8")
        md.write_text(existing + md_block + "\n\n", encoding="utf-8")
        print("✅ Añadido →", md)

def main():
    ap = argparse.ArgumentParser(description="Convierte un PDF en Markdown secuencialmente usando GPT-4 Vision")
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
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = Path.cwd()
    (base / "pages").mkdir(exist_ok=True)
    (base / "images").mkdir(exist_ok=True)

    nombre_base = pdf_path.stem
    md_file = out_dir / f"{nombre_base}.md"
    md_file.write_text("", encoding="utf-8")

    mini = split_pdf(pdf_path, base / "pages")
    jpgs = pdfs_to_jpgs(mini, base / "images", args.dpi)
    process(jpgs, md_file)

    print(f"\n🎉 Fin. Markdown generado: {md_file}")

if __name__=="__main__":
    main()

