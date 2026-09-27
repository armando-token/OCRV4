#!/usr/bin/env python3
"""
pdf_to_markdown_seq.py - Sequential PDF to Markdown OCR Pipeline using Google Gemini.

Extracts pages from a PDF document, converts them into high-resolution images,
and performs OCR via Google Gemini Vision API to assemble a complete Markdown document.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
import google.generativeai as genai
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from PIL import Image
from PyPDF2 import PdfReader, PdfWriter
from tqdm import tqdm

# Load environment variables from .env file if present
load_dotenv()

# Default Gemini model configured via environment variable or default fallback
DEFAULT_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

POPPLER_ERROR_MESSAGE: str = (
    "Poppler not found. Install with: brew install poppler (macOS) or "
    "apt install poppler-utils (Linux)"
)


def get_gemini_api_key() -> Optional[str]:
    """Retrieve the Gemini API key from environment variables.

    Checks GEMINI_API_KEY first, falling back to GOOGLE_API_KEY for legacy support.

    Returns:
        Optional[str]: Found API key, or None if neither environment variable is set.
    """
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def resolve_pdf_path(pdf_arg: str, input_dir: Path) -> Path:
    """Resolve the PDF path either from direct path or within input_dir.

    Args:
        pdf_arg: Direct file path or filename inside input_dir.
        input_dir: Directory containing input files.

    Returns:
        Path: Resolved absolute path to the PDF file.

    Raises:
        FileNotFoundError: If the PDF cannot be located.
    """
    direct_path = Path(pdf_arg)
    if direct_path.is_file():
        return direct_path.resolve()

    in_input_path = input_dir / pdf_arg
    if in_input_path.is_file():
        return in_input_path.resolve()

    # Fallback: check if appending .pdf matches a file
    if not pdf_arg.lower().endswith(".pdf"):
        direct_with_ext = Path(f"{pdf_arg}.pdf")
        if direct_with_ext.is_file():
            return direct_with_ext.resolve()

        candidate_with_ext = input_dir / f"{pdf_arg}.pdf"
        if candidate_with_ext.is_file():
            return candidate_with_ext.resolve()

    raise FileNotFoundError(
        f"PDF file not found: '{pdf_arg}'.\n"
        f"  Checked direct path: {direct_path.resolve()}\n"
        f"  Checked input directory: {in_input_path.resolve()}"
    )


def format_page_number(page_number: int, total_pages: int) -> str:
    """Format a page number with zero padding matching the total page count.

    Args:
        page_number: Current 1-based page index.
        total_pages: Total number of pages.

    Returns:
        str: Zero-padded page number string.
    """
    return str(page_number).zfill(len(str(total_pages)))


def split_pdf(pdf_path: Path, temp_dir: Path) -> List[Path]:
    """Split a multi-page PDF into individual single-page PDF files.

    Args:
        pdf_path: Path to the source PDF.
        temp_dir: Directory to save individual page files.

    Returns:
        List[Path]: List of paths to the single-page PDF files.
    """
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    page_paths: List[Path] = []

    for idx in tqdm(range(total_pages), desc="Splitting PDF", unit="page"):
        writer = PdfWriter()
        writer.add_page(reader.pages[idx])
        page_file = temp_dir / f"{format_page_number(idx + 1, total_pages)}.pdf"
        with page_file.open("wb") as f_out:
            writer.write(f_out)
        page_paths.append(page_file)

    return page_paths


def pdfs_to_jpgs(page_pdfs: List[Path], temp_dir: Path, dpi: int) -> List[Path]:
    """Convert single-page PDFs into high-resolution JPEG images.

    Args:
        page_pdfs: List of paths to single-page PDF files.
        temp_dir: Directory where JPEG images will be stored.
        dpi: Target resolution in dots per inch.

    Returns:
        List[Path]: Paths to generated JPEG images.

    Raises:
        RuntimeError: If Poppler is missing or page conversion fails.
    """
    image_paths: List[Path] = []

    for page_pdf in tqdm(page_pdfs, desc="Converting PDF to JPG", unit="page"):
        try:
            pil_images = convert_from_path(str(page_pdf), dpi=dpi, fmt="jpeg")
        except PDFInfoNotInstalledError as exc:
            raise RuntimeError(POPPLER_ERROR_MESSAGE) from exc
        except Exception as exc:
            error_str = str(exc).lower()
            if any(term in error_str for term in ("poppler", "pdfinfo", "pdftoppm")):
                raise RuntimeError(POPPLER_ERROR_MESSAGE) from exc
            print(f"[!] Warning: Failed to convert {page_pdf.name} to JPG: {exc}", file=sys.stderr)
            continue

        if pil_images:
            jpg_path = temp_dir / page_pdf.with_suffix(".jpg").name
            pil_images[0].save(jpg_path, "JPEG", quality=95)
            image_paths.append(jpg_path)
        else:
            print(f"[!] Warning: No image output generated for {page_pdf.name}", file=sys.stderr)

    return image_paths


def ocr_gemini(
    image_path: Path,
    model_name: str,
    max_retries: int = 3,
    backoff: float = 2.0,
) -> Optional[str]:
    """Extract text from a single image using Google Gemini Vision API.

    Args:
        image_path: Path to the page image file.
        model_name: Gemini model identifier.
        max_retries: Number of retry attempts on transient network or API errors.
        backoff: Base multiplier for exponential backoff between retries.

    Returns:
        Optional[str]: Extracted text in Markdown format, or None if extraction failed.
    """
    img = Image.open(image_path)
    prompt_parts = [
        (
            "Extract all visible text from this image. "
            "Output only the extracted text formatted cleanly in Markdown, "
            "without any commentary, preamble, or surrounding conversational text. "
            "Preserve formatting, tables, lists, and line breaks as accurately as possible."
        ),
        img,
    ]
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=8192,
    )
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        safety_settings=safety_settings,
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt_parts)
            if hasattr(response, "text") and response.text:
                return response.text
            if response.candidates and response.candidates[0].content.parts:
                parts_text = "".join(
                    part.text
                    for part in response.candidates[0].content.parts
                    if hasattr(part, "text") and part.text
                )
                if parts_text:
                    return parts_text
            print(f"[!] Warning: No text returned for {image_path.name} (attempt {attempt}/{max_retries}).")
        except Exception as exc:
            if attempt < max_retries:
                sleep_duration = backoff * attempt
                print(
                    f"[-] Gemini request failed for {image_path.name} "
                    f"(attempt {attempt}/{max_retries}): {exc}. "
                    f"Retrying in {sleep_duration:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(sleep_duration)
            else:
                print(
                    f"[x] Error: Final OCR failure for {image_path.name} "
                    f"after {max_retries} attempts: {exc}",
                    file=sys.stderr,
                )
                return None

    return None


def process_images_to_markdown(
    image_paths: List[Path],
    model_name: str,
) -> str:
    """Process a list of page images through Gemini OCR and assemble into Markdown.

    Args:
        image_paths: List of page image paths.
        model_name: Gemini model identifier.

    Returns:
        str: Consolidated Markdown content for all pages.
    """
    extracted_sections: List[str] = []
    total_images = len(image_paths)

    print(f"\n[+] Starting OCR extraction with model '{model_name}' ({total_images} pages)...")
    for idx, img_path in enumerate(tqdm(image_paths, desc="OCR with Gemini", unit="page"), start=1):
        try:
            page_text = ocr_gemini(img_path, model_name=model_name)
            if page_text and page_text.strip():
                extracted_sections.append(page_text.strip())
            else:
                print(f"[!] Warning: No text extracted from page {idx} ({img_path.name}).")
        except Exception as exc:
            print(f"[x] Error processing page {idx} ({img_path.name}): {exc}", file=sys.stderr)

    if not extracted_sections:
        print("[!] Warning: No text extracted from any page.")
        return ""

    print(f"[✓] Extracted text from {len(extracted_sections)}/{total_images} pages.")
    return "\n\n".join(extracted_sections)


def parse_arguments() -> argparse.Namespace:
    """Configure and parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert a PDF into Markdown using Google Gemini OCR."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the PDF file (absolute/relative path or filename inside --input-dir).",
    )
    parser.add_argument(
        "--input-dir",
        default="input",
        help="Input directory containing PDF files (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for generated Markdown files (default: %(default)s).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rendering resolution in DPI for PDF to JPG conversion (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model name for OCR (default: from GEMINI_MODEL env or %(default)s).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Retain temporary page and image directories for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    """Main CLI execution flow."""
    args = parse_arguments()

    # Validate Gemini API credentials
    api_key = get_gemini_api_key()
    if not api_key:
        print(
            "[x] Error: Gemini API key not found.\n"
            "Please set GEMINI_API_KEY (or GOOGLE_API_KEY) in your .env file or environment variables.",
            file=sys.stderr,
        )
        sys.exit(1)

    genai.configure(api_key=api_key)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    try:
        pdf_path = resolve_pdf_path(args.pdf, input_dir)
    except FileNotFoundError as exc:
        print(f"[x] Error: {exc}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = pdf_path.stem
    output_md_path = output_dir / f"{base_name}.md"

    # Temporary working directories
    workspace = Path.cwd()
    pages_dir = workspace / "pages_temp"
    images_dir = workspace / "images_temp"
    pages_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Input PDF: {pdf_path}")
    print(f"[*] Output Markdown: {output_md_path}")
    print(f"[*] Model: {args.model}")
    print(f"[*] DPI: {args.dpi}")

    try:
        # Step 1: Split PDF into individual pages
        page_pdfs = split_pdf(pdf_path, pages_dir)
        if not page_pdfs:
            print("[x] Error: Unable to split PDF pages.", file=sys.stderr)
            sys.exit(1)

        # Step 2: Convert individual PDF pages to JPG images
        page_images = pdfs_to_jpgs(page_pdfs, images_dir, args.dpi)
        if not page_images:
            print("[x] Error: Unable to convert PDF pages to images.", file=sys.stderr)
            sys.exit(1)

        # Step 3: Run Gemini OCR on extracted images
        markdown_text = process_images_to_markdown(page_images, model_name=args.model)

        # Step 4: Write result to output Markdown file
        if markdown_text:
            output_md_path.write_text(markdown_text + "\n", encoding="utf-8")
            print(f"\n[✓] Markdown successfully saved: {output_md_path}")
        else:
            print(
                "\n[!] Warning: No markdown generated (OCR returned empty or failed for all pages).",
                file=sys.stderr,
            )

    except RuntimeError as exc:
        print(f"[x] Execution Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"[x] Unexpected Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if args.keep_temp:
            print(f"[*] Keeping temporary directories:\n  - {pages_dir}\n  - {images_dir}")
        else:
            for temp_path in (pages_dir, images_dir):
                if temp_path.exists():
                    try:
                        shutil.rmtree(temp_path)
                    except Exception as exc:
                        print(f"[!] Warning: Could not remove {temp_path}: {exc}", file=sys.stderr)
            print("[*] Temporary directories cleaned up.")


if __name__ == "__main__":
    main()
