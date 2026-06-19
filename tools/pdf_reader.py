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

def run_sarvam_ocr(file_path: str, api_key: str, language: str = "hi-IN") -> List[str]:
    import zipfile
    import json
    import tempfile
    from pathlib import Path
    from sarvamai import SarvamAI
    
    # Initialize the client
    client = SarvamAI(api_subscription_key=api_key)
    
    # 1. Create a job
    job = client.document_intelligence.create_job(
        language=language,     # BCP-47 language code
        output_format="md"    # "md" or "html" — NOT "markdown"
    )
    
    # 2. Upload your document
    job.upload_file("data/Patient Data.pdf")
    
    # 3. Start processing
    job.start()
    
    # 4. Wait for completion
    status = job.wait_until_complete()
    print(f"Status: {status.job_state}")
    
    # 5. Download output (a ZIP with your MD/HTML + JSON)
    try:
        job.download_output("./output.zip")
        print("Downloaded output to ./output.zip")
    except Exception as exc:
        print(f"Warning: Could not download output to ./output.zip: {exc}")
        
    pages_text = []
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = Path(temp_dir) / "output.zip"
        job.download_output(str(zip_path))
        
        # 4. Extract and read
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all metadata files
            meta_files = sorted([n for n in zip_ref.namelist() if n.startswith("metadata/page_") and n.endswith(".json")])
            
            for meta_file in meta_files:
                with zip_ref.open(meta_file) as f:
                    page_data = json.loads(f.read().decode('utf-8'))
                    blocks = page_data.get("blocks", [])
                    # Sort blocks by reading_order
                    blocks.sort(key=lambda x: x.get("reading_order", 0))
                    page_text = "\n\n".join(b.get("text", "") for b in blocks if b.get("text"))
                    pages_text.append(page_text.strip())
                    
    return pages_text

def run_sarvam_ocr_chunked(file_path: str, api_key: str, page_limit: int = 100, language: str = "hi-IN") -> List[str]:
    import fitz
    import tempfile
    import os
    
    try:
        src_doc = fitz.open(file_path)
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF file {file_path} for chunking: {exc}")
        
    total_pages = len(src_doc)
    pages_to_read = min(total_pages, page_limit)
    pages_text = ["" for _ in range(pages_to_read)]
    
    chunk_size = 10
    
    for start_page in range(0, pages_to_read, chunk_size):
        end_page = min(start_page + chunk_size, pages_to_read)
        print(f"Processing pages {start_page+1} to {end_page} with Sarvam OCR...")
        
        # Create a temp PDF for the chunk
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_pdf_path = temp_file.name
            
        try:
            # Save the page subset to the temp PDF
            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(src_doc, from_page=start_page, to_page=end_page-1)
            chunk_doc.save(temp_pdf_path)
            chunk_doc.close()
            
            # Run Sarvam OCR on the chunk
            chunk_texts = run_sarvam_ocr(temp_pdf_path, api_key, language=language)
            
            # Map chunk outputs back to correct absolute page indices
            for offset, text in enumerate(chunk_texts):
                abs_page = start_page + offset
                if abs_page < pages_to_read:
                    pages_text[abs_page] = text
                    
        except Exception as exc:
            print(f"Error processing chunk {start_page+1}-{end_page}: {exc}")
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
                
    src_doc.close()
    return pages_text

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
        # Target the specific patient document to avoid scanning .venv or other directories
        pdf_files = sorted(target.glob("discharge_agent/data/Patient Data.pdf"))
        if not pdf_files:
            pdf_files = sorted(target.glob("**/Patient Data.pdf"))
        
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
    
    # Check if the PDF has a selectable text layer
    has_text = False
    total_extracted_chars = 0
    for page_num in range(min(5, pages_to_read)):
        page = doc[page_num]
        total_extracted_chars += len(page.get_text("text").strip())
    if total_extracted_chars > 200:
        has_text = True

    # If it is a scanned PDF (no selectable text) and Sarvam API Key is available, run Sarvam OCR
    SARVAMAI_API_KEY = os.environ.get("SARVAMAI_API_KEY", "")
    SARVAM_LANGUAGE = os.environ.get("SARVAM_LANGUAGE", "hi-IN")
    if not has_text and SARVAMAI_API_KEY:
        print(f"Scanned PDF detected. Using Sarvam AI Document Intelligence for OCR: {file_path}")
        try:
            sarvam_pages = run_sarvam_ocr_chunked(
                file_path, 
                SARVAMAI_API_KEY, 
                page_limit=pages_to_read, 
                language=SARVAM_LANGUAGE
            )
            # Build docs from Sarvam output
            for page_num in range(pages_to_read):
                text = ""
                if page_num < len(sarvam_pages):
                    text = sarvam_pages[page_num]
                
                docs.append(
                    PatientDocument(
                        file_path=file_path,
                        page_number=page_num + 1,
                        note_type="unknown",
                        raw_text=text.strip(),
                        extraction_confidence=0.95 if text.strip() else 0.1,
                        read_error=None if text.strip() else "Empty page or unreadable",
                    )
                )
            doc.close()
            return docs
        except Exception as exc:
            print(f"Sarvam AI OCR failed: {exc}. Falling back to standard PyMuPDF/Gemini reader.")

    # Fall back to standard PyMuPDF / Gemini OCR
    for page_num in range(pages_to_read):
        try:
            page = doc[page_num]
            text = page.get_text("text")
            confidence = 1.0
            if len(text.strip()) < 50:
                blocks = page.get_text("blocks")
                text = " ".join(b[4] for b in blocks if isinstance(b[4], str))
                confidence = 0.4 if text.strip() else 0.1
                
            # Perform Gemini OCR fallback if page has no selectable text and fallback is enabled
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
