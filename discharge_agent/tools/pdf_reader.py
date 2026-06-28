from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

# Support executing this script directly
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from config.settings import SETTINGS
from models.patient import PatientDocument
from tools.base import BaseTool, ToolResult, ToolStatus


class PDFReaderTool(BaseTool):
    name = "pdf_reader"
    description = (
        "Extract text from one or all PDFs in a directory. "
        "Returns a list of PatientDocument objects, one per page. "
        "Pages that cannot be read are marked with read_error."
    )

    def _run(
        self,
        path: str,
        page_limit: Optional[int] = None,
    ) -> ToolResult:
        try:
            import fitz
        except ImportError:
            return ToolResult(
                status=ToolStatus.FAILED,
                error="PyMuPDF (fitz) not installed. Run: pip install pymupdf",
            )

        page_limit = page_limit or SETTINGS.max_pdf_pages
        target = Path(path)

        if target.is_dir():
            pdf_files = sorted(target.glob("discharge_agent/data/Medical bills.pdf"))
            if not pdf_files:
                pdf_files = sorted(target.glob("**/Medical bills.pdf"))
            if not pdf_files:
                return ToolResult(
                    status=ToolStatus.NOT_FOUND,
                    error=f"No PDF files found in directory: {path}",
                )
        elif target.is_file() and target.suffix.lower() == ".pdf":
            pdf_files = [target]
        else:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                error=f"Path is not a PDF or directory: {path}",
            )

        all_docs: List[PatientDocument] = []
        unreadable: List[str] = []

        for pdf_path in pdf_files:
            docs, errors = self._extract_pdf(str(pdf_path), fitz, page_limit)
            all_docs.extend(docs)
            unreadable.extend(errors)

        if not all_docs:
            return ToolResult(
                status=ToolStatus.FAILED,
                error=f"Could not extract text from any PDF. Unreadable: {unreadable}",
                data={"unreadable": unreadable},
            )

        status = ToolStatus.SUCCESS if not unreadable else ToolStatus.PARTIAL
        return ToolResult(
            status=status,
            data=all_docs,
            metadata={
                "total_pages": len(all_docs),
                "unreadable_files": unreadable,
                "pdf_count": len(pdf_files),
            },
            error=f"Some files unreadable: {unreadable}" if unreadable else None,
        )

    def _extract_pdf(
        self,
        pdf_path: str,
        fitz_module,
        page_limit: int,
    ):
        docs: List[PatientDocument] = []
        errors: List[str] = []

        try:
            doc = fitz_module.open(pdf_path)
        except Exception as exc:
            errors.append(f"{pdf_path}: {exc}")
            return docs, errors

        pages_to_read = min(len(doc), page_limit)

        for page_num in range(pages_to_read):
            try:
                page = doc[page_num]
                text = page.get_text("text")

                # If text layer is nearly empty, flag low confidence (likely scanned/handwritten)
                confidence = 1.0
                if len(text.strip()) < 50:
                    # Try blocks extraction as fallback
                    blocks = page.get_text("blocks")
                    text = " ".join(b[4] for b in blocks if isinstance(b[4], str))
                    confidence = 0.4 if text.strip() else 0.1

                patient_doc = PatientDocument(
                    file_path=pdf_path,
                    page_number=page_num + 1,
                    note_type="unknown",  # Classified later by DocumentParserTool
                    raw_text=text.strip(),
                    extraction_confidence=confidence,
                    read_error=None if text.strip() else "Empty page or unreadable",
                )
                docs.append(patient_doc)

            except Exception as exc:
                # Page-level failure — mark and continue
                docs.append(
                    PatientDocument(
                        file_path=pdf_path,
                        page_number=page_num + 1,
                        note_type="unknown",
                        raw_text="",
                        extraction_confidence=0.0,
                        read_error=f"Page extraction failed: {exc}",
                    )
                )

        doc.close()
        return docs, errors

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "path": {
                    "type": "string",
                    "description": "File path to a single PDF or a directory containing PDFs",
                    "required": True,
                },
                "page_limit": {
                    "type": "integer",
                    "description": f"Max pages per PDF (default {SETTINGS.max_pdf_pages})",
                    "required": False,
                },
            },
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test PDF extraction.")
    parser.add_argument("pdf_path", help="Path to PDF file or directory of PDFs")
    args = parser.parse_args()

    tool = PDFReaderTool()
    result = tool.run(path=args.pdf_path)
    if result.ok:
        print(f"Successfully extracted {result.metadata['total_pages']} page(s).")
        for doc in result.data:
            print(f"\n--- Page {doc.page_number} ({doc.file_path}) ---")
            print(doc.raw_text[:500])
            print("...")
    else:
        print(f"Error: {result.error}")
