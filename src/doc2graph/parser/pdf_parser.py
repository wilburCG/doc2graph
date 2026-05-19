"""PDF document parser with OCR support and page selection."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Callable, Sequence

from doc2graph.parser.base import BaseParser

logger = logging.getLogger(__name__)

# Progress callback type: receives dict with phase, page_current, page_total, message
ProgressCallback = Callable[[dict], None] | None


def parse_page_range(page_spec: str | None, total_pages: int) -> list[int]:
    """Parse a page range specification into a list of 0-based page indices.

    Args:
        page_spec: Page range string like "1,3,5-10" or "all" or None.
                   Uses 1-based page numbers (user-facing).
        total_pages: Total number of pages in the document.

    Returns:
        Sorted list of unique 0-based page indices.
    """
    if page_spec is None or page_spec.strip().lower() in ("all", ""):
        return list(range(total_pages))

    pages = set()
    for part in page_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            # Convert 1-based to 0-based, clamp to valid range
            start = max(1, min(start, total_pages))
            end = max(start, min(end, total_pages))
            pages.update(range(start - 1, end))
        else:
            page_num = int(part)
            if 1 <= page_num <= total_pages:
                pages.add(page_num - 1)

    return sorted(pages)


class PdfParser(BaseParser):
    """Parse PDF files with OCR support and selective page processing."""

    def parse(
        self,
        file_path: str | Path,
        pages: str | None = None,
        ocr_enabled: bool = True,
        ocr_lang: str = "chi_sim+eng",
        progress_callback: ProgressCallback = None,
    ) -> str:
        """Parse a PDF file, optionally extracting specific pages and using OCR.

        Args:
            file_path: Path to the PDF file.
            pages: Page range specification (e.g. "1,3,5-10", "1", "all").
                   None or "all" means all pages. Uses 1-based numbering.
            ocr_enabled: Whether to use OCR for pages with images.
            ocr_lang: Tesseract language pack (e.g. "chi_sim+eng").
            progress_callback: Optional callback for progress reporting.

        Returns:
            Extracted text content.
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"PDF file not found: {p}")

        # Try PyMuPDF first (has built-in OCR + image extraction)
        try:
            return self._parse_with_mupdf(p, pages, ocr_enabled, ocr_lang, progress_callback)
        except ImportError:
            pass

        return self._parse_with_pypdf2(p, pages)

    def get_page_count(self, file_path: str | Path) -> int:
        """Get total number of pages in the PDF."""
        p = Path(file_path)
        try:
            import fitz
            doc = fitz.open(str(p))
            count = len(doc)
            doc.close()
            return count
        except ImportError:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(p))
            return len(reader.pages)

    def _parse_with_mupdf(
        self,
        p: Path,
        page_spec: str | None,
        ocr_enabled: bool,
        ocr_lang: str,
        progress_callback: ProgressCallback = None,
    ) -> str:
        import fitz  # pymupdf

        doc = fitz.open(str(p))
        total_pages = len(doc)
        page_indices = parse_page_range(page_spec, total_pages)

        if page_spec and page_spec.strip().lower() not in ("all", ""):
            logger.info(f"Processing pages {page_spec} (0-based: {page_indices}) of {total_pages}")

        texts = []
        total = len(page_indices)
        for i, page_num in enumerate(page_indices):
            page = doc[page_num]

            # Extract text
            text = page.get_text().strip()

            # OCR for images if enabled
            if ocr_enabled:
                image_text = self._ocr_page_images(page, page_num, ocr_lang)
                if image_text:
                    if text:
                        text = text + "\n\n[OCR Extracted]\n" + image_text
                    else:
                        text = "[OCR Extracted]\n" + image_text

            page_label = f"--- Page {page_num + 1} ---"
            if text:
                texts.append(f"{page_label}\n{text}")
            else:
                texts.append(f"{page_label}\n(No text content)")

            # Report progress
            if progress_callback:
                progress_callback({
                    "phase": "parsing",
                    "page_current": i + 1,       # index in selected pages (1-based)
                    "page_total": total,          # total selected pages
                    "page_absolute": page_num + 1, # absolute page number in PDF
                    "message": f"正在解析第 {page_num + 1} 页",
                })

        doc.close()
        return "\n\n".join(texts)

    def _ocr_page_images(self, page, page_num: int, ocr_lang: str) -> str:
        """Extract and OCR images from a PDF page.

        Optimizations:
        - Upscale images below 300 DPI for better OCR accuracy
        - Set explicit DPI to avoid Tesseract "Invalid resolution" warnings
        - Skip images that are too small or unlikely to contain text
        """
        import pytesseract
        from PIL import Image, ImageFilter

        image_texts = []
        try:
            images = page.get_images(full=True)
            if not images:
                return ""

            for img_idx, img_info in enumerate(images):
                xref = img_info[0]
                base_image = page.parent.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Skip very small images (likely icons/decorations)
                if len(image_bytes) < 2000:
                    continue

                try:
                    image = Image.open(io.BytesIO(image_bytes))

                    # Skip tiny images
                    if image.width < 100 or image.height < 50:
                        continue

                    # Optimization: upscale low-resolution images
                    # Tesseract works best at 300+ DPI
                    scale_factor = max(1, 300 / max(image.width, image.height) if max(image.width, image.height) < 300 else 1)
                    if scale_factor > 1:
                        new_width = int(image.width * scale_factor)
                        new_height = int(image.height * scale_factor)
                        image = image.resize((new_width, new_height), Image.LANCZOS)
                        logger.debug(f"Upscaled image {img_idx + 1} on page {page_num + 1} by {scale_factor:.1f}x")

                    # Set explicit DPI to avoid Tesseract "Invalid resolution 0 dpi" warnings
                    image.info["dpi"] = (300, 300)

                    # Convert to grayscale for better OCR
                    if image.mode != "L":
                        image = image.convert("L")

                    # Apply mild sharpening
                    image = image.filter(ImageFilter.SHARPEN)

                    # OCR with PSM auto for better layout detection
                    custom_config = "--psm 6"  # Assume uniform block of text
                    ocr_text = pytesseract.image_to_string(image, lang=ocr_lang, config=custom_config).strip()

                    if ocr_text and len(ocr_text) > 10:
                        image_texts.append(f"[Image {img_idx + 1} on page {page_num + 1}]\n{ocr_text}")
                        logger.info(f"OCR extracted {len(ocr_text)} chars from image {img_idx + 1} on page {page_num + 1}")
                except Exception as e:
                    # Log at debug level to avoid noise from non-text images
                    logger.debug(f"OCR skipped for image {img_idx + 1} on page {page_num + 1}: {e}")

        except Exception as e:
            logger.warning(f"Image extraction failed for page {page_num + 1}: {e}")

        return "\n\n".join(image_texts)

    def _parse_with_pypdf2(self, p: Path, page_spec: str | None) -> str:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(p))
        total_pages = len(reader.pages)
        page_indices = parse_page_range(page_spec, total_pages)

        texts = []
        for page_num in page_indices:
            page = reader.pages[page_num]
            text = page.extract_text()
            if text:
                texts.append(f"--- Page {page_num + 1} ---\n{text}")

        return "\n\n".join(texts)

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return [".pdf"]
