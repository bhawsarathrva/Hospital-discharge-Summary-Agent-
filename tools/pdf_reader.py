"""
tools/pdf_reader.py
PDF ingestion tool using PyMuPDF (fitz).
"""
from __future__ import annotations
import sys
import os
import fitz
from pathlib import Path
from typing import List, Optional

workspace_root = str(Path(__file__).resolve().parent.parent)
discharge_agent_dir = str(Path(__file__).resolve().parent.parent / "discharge_agent")
if discharge_agent_dir not in sys.path:
    sys.path.insert(0, discharge_agent_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from dotenv import load_dotenv
from discharge_agent.config.settings import SETTINGS
from discharge_agent.models.patient import PatientDocument

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")

def read_pdf_directory(directory_path: str, page_limit: int = 100) -> List[PatientDocument]:
    """Read all PDF documents in a directory or a single PDF file."""
    target = Path(directory_path)
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {directory_path}")
        
    if target.is_file():
        if target.suffix.lower() == ".pdf":
            pdf_files = [target]
        else:
            raise ValueError(f"Path is not a PDF file: {directory_path}")
    else:
        # Support finding any PDFs in the directory recursively
        pdf_files = sorted(target.glob("**/*.pdf"))
        
    all_docs = []
    for pdf in pdf_files:
        all_docs.extend(read_pdf_file(str(pdf), page_limit))
    return all_docs

def read_pdf_file(file_path: str, page_limit: int = 100) -> List[PatientDocument]:
    """Read a single PDF document and return a list of PatientDocuments (one per page)."""
    docs = []
    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF file {file_path}: {exc}")
        
    pages_to_read = min(len(doc), page_limit)
    for page_num in range(pages_to_read):
        try:
            page = doc[page_num]
            text = page.get_text("text")
            confidence = 1.0
            if len(text.strip()) < 50:
                blocks = page.get_text("blocks")
                text = " ".join(b[4] for b in blocks if isinstance(b[4], str))
                confidence = 0.4 if text.strip() else 0.1
                
            # Perform OCR fallback if page has no selectable text and fallback is enabled
            if len(text.strip()) < 50 and getattr(SETTINGS, "ocr_fallback", True) and GOOGLE_API_KEY:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=GOOGLE_API_KEY)
                    ocr_model = genai.GenerativeModel("gemini-1.5-flash")
                    
                    # Render page as PNG image at 150 DPI for legibility
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    
                    response = ocr_model.generate_content([
                        {"mime_type": "image/png", "data": img_bytes},
                        "Transcribe all handwritten and printed text from this medical record page verbatim. Do not summarize or interpret, just return the raw text."
                    ])
                    if response.text:
                        text = response.text
                        confidence = 0.95
                except Exception as exc:
                    print(f"OCR Error page {page_num+1}: {exc}")
                
            docs.append(
                PatientDocument(
                    file_path=file_path,
                    page_number=page_num + 1,
                    note_type="unknown",
                    raw_text=text.strip(),
                    extraction_confidence=confidence,
                    read_error=None if text.strip() else "Empty page or unreadable",
                )
            )
        except Exception as exc:
            docs.append(
                PatientDocument(
                    file_path=file_path,
                    page_number=page_num + 1,
                    note_type="unknown",
                    raw_text="",
                    extraction_confidence=0.0,
                    read_error=str(exc),
                )
            )
    doc.close()
    return docs
