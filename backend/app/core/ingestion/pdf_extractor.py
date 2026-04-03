"""
core/ingestion/pdf_extractor.py — PDF text extraction via PyMuPDF

Why PyMuPDF (fitz)?
    - Fastest Python PDF library; handles most PDF variants including
      scanned documents with embedded text layers.
    - Extracts text in natural reading order per page.
    - Preserves page number metadata, which we surface in citations.

Limitations (demo scope):
    - Does not perform OCR on image-only PDFs (scanned without text layer).
    - Table structure is extracted as flat text (no structured table parsing).
    Production upgrade: swap in Unstructured.io or Azure Document Intelligence
    for OCR, table detection, and layout-aware extraction.

Output:
    A list of PageText objects, one per PDF page, each carrying the raw text
    and its 1-based page number.
"""

from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class PageText:
    """
    Raw text extracted from a single PDF page.

    Attributes:
        page_number: 1-based page index (matches what users see in a PDF viewer).
        text:        Extracted plain text for this page.  May contain line-break
                     artefacts from the PDF layout engine.
    """

    page_number: int
    text: str


def extract_pages(file_bytes: bytes) -> list[PageText]:
    """
    Extract text from every page of a PDF given its raw bytes.

    Args:
        file_bytes: Raw bytes of the PDF file (as received from the upload).

    Returns:
        List of PageText objects in page order.  Pages with no extractable
        text (e.g. blank pages or pure images) are included with empty text
        so that page numbering stays consistent.

    Raises:
        ValueError: If the bytes cannot be parsed as a valid PDF.
    """
    try:
        # Open from bytes — avoids writing a temp file to disk.
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Failed to open PDF: {exc}") from exc

    pages: list[PageText] = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        # "text" sort preserves reading order; "blocks" gives bounding boxes
        # (not needed here, but useful if adding table support later).
        raw_text = page.get_text("text")
        pages.append(
            PageText(
                page_number=page_index + 1,  # convert 0-based index to 1-based
                text=raw_text,
            )
        )

    doc.close()
    return pages
