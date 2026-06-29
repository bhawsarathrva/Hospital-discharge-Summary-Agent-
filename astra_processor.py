import argparse
import os
import sys
import re
import json
from pathlib import Path
from dotenv import load_dotenv

# Reconfigure stdout/stderr to UTF-8 to prevent UnicodeEncodeErrors on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from astrapy import DataAPIClient
from astrapy.constants import VectorMetric
from astrapy.info import CollectionDefinition

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


def extract_metadata_regex(text: str):
    """Fallback regex-based metadata extraction for patients/bills."""
    patient_name = "Unknown Patient"
    patient_id = "Unknown ID"
    final_bill = 0.0
    disease = "Unknown Disease"
    
    # Patient Name
    name_match = re.search(r"Patient\s*Name:\s*\n*([^\n\r]+)(?:\n([^\n\r]+))?", text, re.IGNORECASE)
    if name_match:
        p1 = name_match.group(1).strip()
        p2 = name_match.group(2).strip() if name_match.group(2) else ""
        if p2 and "UHID" not in p2 and "Patient ID" not in p2 and "Age" not in p2 and "Gender" not in p2:
            patient_name = f"{p1} {p2}".strip()
        else:
            patient_name = p1
            
    # Patient ID / UHID
    id_match = re.search(r"(?:UHID\s*/\s*Patient\s*ID|Patient\s*ID|UHID):\s*\n*([^\n\r]+)", text, re.IGNORECASE)
    if id_match:
        patient_id = id_match.group(1).strip()
        
    # Diagnosis / Disease
    disease_match = re.search(r"(?:Diagnosis|Principal\s*Diagnosis):\s*\n*([^\n\r]+)(?:\n([^\n\r]+))?", text, re.IGNORECASE)
    if disease_match:
        d1 = disease_match.group(1).strip()
        d2 = disease_match.group(2).strip() if disease_match.group(2) else ""
        if d2 and "ICD-10" not in d2 and "Code" not in d2:
            disease = f"{d1} {d2}".strip()
        else:
            disease = d1
            
    # Gross Total / Billing Summary Total (ignore sub-totals and find last match in text)
    bill_matches = list(re.finditer(r"(?<!Sub-)(?<!Sub\s)(?:\bGROSS\s*TOTAL\b|\bGROSS\b|\bTOTAL\s*BILL\s*AMOUNT\b|\bTOTAL\b)[:\s]*\n*(?:[^\n\r]*?Amount[^\n\r]*?\n)?(?:[^\d\n\r]*?)(\d[\d,.]*)", text, re.IGNORECASE))
    if bill_matches:
        bill_str = bill_matches[-1].group(1).replace(",", "")
        try:
            final_bill = float(bill_str)
        except ValueError:
            pass
            
    # Clean up fields to strip trailing layout headers/artifacts
    patient_name = re.split(r"(?:UHID|Patient\s*ID|Age|Gender|Ward|Room|Date|Treating|Doctor|Department)", patient_name, flags=re.IGNORECASE)[0].strip()
    patient_name = re.sub(r"\s*[/\\-]?\s*(?:UHID|Patient\s*ID).*$", "", patient_name, flags=re.IGNORECASE).strip()
    
    patient_id = re.split(r"(?:Age|Gender|Ward|Room|Date|Diagnosis|Treating|Doctor|Department)", patient_id, flags=re.IGNORECASE)[0].strip()
    
    disease = re.split(r"(?:ICD-10|Code|Insurance|TPA|Policy|Amount|BED\s*&|ACCOMMODATION)", disease, flags=re.IGNORECASE)[0].strip()
            
    return {
        "patient_name": patient_name,
        "patient_id": patient_id,
        "final_bill": final_bill,
        "disease": disease
    }


def extract_pdf_metadata(full_text: str):
    """Extract patient name, patient id, final bill, and disease via Gemini LLM (fallback to Regex)."""
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("Warning: GOOGLE_API_KEY not found in environment. Using regex fallback for metadata.")
        return extract_metadata_regex(full_text)
        
    try:
        import google.generativeai as genai
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are a medical record processing assistant. Analyze the clinical/billing text below and extract these fields:
        1. Patient Name (cleanly format full name, e.g. "Mr. Rajesh Kumar Sharma")
        2. Patient ID (often labelled as UHID, Patient ID, or UHID / Patient ID)
        3. Final Bill (the gross total amount or final billing amount in the document, e.g. 155700, 108900)
        4. Disease (the principal diagnosis or disease being treated, e.g. Acute Myocardial Infarction, Type 2 Diabetes Mellitus)

        Provide the output as a valid JSON object with the keys "patient_name", "patient_id", "final_bill", and "disease".
        Make sure your output is ONLY the JSON object, formatted like:
        {{
          "patient_name": "extracted name",
          "patient_id": "extracted ID",
          "final_bill": 123450.0,
          "disease": "extracted diagnosis"
        }}
        If any field is not found, set its value to null. Do not include markdown code block characters (like ```json) in your final response.

        Medical/Billing Text:
        {full_text[:30000]}
        """
        
        response = model.generate_content(prompt)
        text_response = response.text.strip()
        if text_response.startswith("```json"):
            text_response = text_response[7:]
        if text_response.endswith("```"):
            text_response = text_response[:-3]
            
        data = json.loads(text_response.strip())
        return {
            "patient_name": data.get("patient_name") or "Unknown Patient",
            "patient_id": data.get("patient_id") or "Unknown ID",
            "final_bill": data.get("final_bill") or 0.0,
            "disease": data.get("disease") or "Unknown Disease"
        }
    except Exception as e:
        print(f"Warning: LLM extraction failed ({e}). Falling back to regex.")
        return extract_metadata_regex(full_text)


def get_db():
    """Load environment variables and return Astra DB database object."""
    load_dotenv()
    
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    token = os.getenv("ASTRA_DB_ACAPPLICATION_TOKEN") or os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    keyspace = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")
    
    if not api_endpoint or not token:
        print("Error: Missing database credentials in environment.", file=sys.stderr)
        print("Please ensure ASTRA_DB_API_ENDPOINT and ASTRA_DB_ACAPPLICATION_TOKEN are set in .env", file=sys.stderr)
        sys.exit(1)
        
    client = DataAPIClient(token)
    database = client.get_database(api_endpoint, keyspace=keyspace)
    return database


def extract_pdf_text(pdf_path: str):
    """Extract text from PDF pages using PyMuPDF (fitz)."""
    print(f"Reading PDF: {pdf_path}")
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    doc = fitz.open(pdf_path)
    pages_data = []
    
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages_data.append({
                "page_number": i + 1,
                "text": text
            })
            
    print(f"Extracted {len(pages_data)} pages from '{path.name}'.")
    return pages_data


def chunk_text(pages_data, chunk_size: int, chunk_overlap: int):
    """Split page text into fixed-size chunks using RecursiveCharacterTextSplitter."""
    print(f"Chunking text (size={chunk_size}, overlap={chunk_overlap})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = []
    for page in pages_data:
        page_num = page["page_number"]
        page_text = page["text"]
        
        split_texts = splitter.split_text(page_text)
        for i, split in enumerate(split_texts):
            chunks.append({
                "text": split,
                "page": page_num,
                "chunk_index": i
            })
            
    print(f"Generated {len(chunks)} text chunks.")
    return chunks


def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    """Compute local embeddings for all chunks using SentenceTransformer."""
    print(f"Loading SentenceTransformer model '{model_name}'...")
    model = SentenceTransformer(model_name)
    
    print("Generating vector embeddings...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    for chunk, emb in zip(chunks, embeddings):
        chunk["vector"] = emb.tolist()
        
    print("Vectorization complete.")
    return chunks


def get_or_create_collection(db, collection_name: str, dimension: int = 384):
    """Retrieve existing collection or create a new vector collection in Astra DB."""
    collections = db.list_collection_names()
    
    if collection_name in collections:
        print(f"Connecting to existing collection: '{collection_name}'")
        return db.get_collection(collection_name)
        
    print(f"Creating new collection: '{collection_name}' (dimension={dimension}, metric=cosine)...")
    collection = db.create_collection(
        collection_name,
        definition=(
            CollectionDefinition.builder()
            .set_vector_dimension(dimension)
            .set_vector_metric(VectorMetric.COSINE)
            .build()
        )
    )
    return collection


def group_pages_by_invoice(pages_data):
    """Group pages by detecting where a new patient record/invoice starts."""
    invoices = []
    current_group = []
    
    for page in pages_data:
        text = page["text"]
        # A new invoice page typically has PATIENT & ADMISSION or UHID / Patient ID
        if "PATIENT & ADMISSION" in text or "Patient Name:" in text or "UHID / Patient ID" in text:
            if current_group:
                invoices.append(current_group)
            current_group = [page]
        else:
            if not current_group:
                current_group = [page]
            else:
                current_group.append(page)
                
    if current_group:
        invoices.append(current_group)
        
    return invoices


def ingest_command(args):
    """Ingest, vectorize, and store PDF content page-by-page (without chunking) into Astra DB with patient metadata."""
    db = get_db()
    
    # 1. Parse PDF
    try:
        pages_data = extract_pdf_text(args.pdf)
    except Exception as e:
        print(f"Error reading PDF: {e}", file=sys.stderr)
        sys.exit(1)
        
    if not pages_data:
        print("No readable text found in PDF. Exiting.", file=sys.stderr)
        sys.exit(1)
        
    # 2. Group pages into invoices to extract patient-specific metadata
    invoice_groups = group_pages_by_invoice(pages_data)
    print(f"Grouped PDF pages into {len(invoice_groups)} patient record/billing groups.")
    
    documents = []
    pdf_name = Path(args.pdf).name
    
    # 3. Connect to/Create collection (384 dimensions for all-MiniLM-L6-v2)
    collection = get_or_create_collection(db, args.collection, dimension=384)
    
    # Clear existing documents for the same PDF to prevent DOCUMENT_ALREADY_EXISTS errors
    try:
        print(f"Clearing existing page records for '{pdf_name}' in collection '{args.collection}'...")
        delete_res = collection.delete_many({"pdf_name": pdf_name})
        print(f"Deleted {delete_res.deleted_count} existing records.")
    except Exception as e:
        print(f"Warning: Failed to clear old records: {e}")
        
    for g_idx, group in enumerate(invoice_groups):
        # Combine text for this patient's invoice group to extract metadata
        group_text = "\n".join([p["text"] for p in group])
        print(f"Extracting metadata for patient group #{g_idx+1} ({len(group)} pages)...")
        metadata = extract_pdf_metadata(group_text)
        
        print(f"\n--- Extracted Metadata Reference for Group #{g_idx+1} ---")
        print(f"  Patient Name : {metadata['patient_name']}")
        print(f"  Patient ID   : {metadata['patient_id']}")
        print(f"  Final Bill   : {metadata['final_bill']}")
        print(f"  Disease      : {metadata['disease']}")
        print("----------------------------------------------------\n")
        
        # Format chunks (pages) for this group
        group_as_chunks = []
        for page in group:
            group_as_chunks.append({
                "text": page["text"],
                "page": page["page_number"],
                "chunk_index": 0
            })
            
        # Vectorize pages in this group
        group_with_vectors = embed_chunks(group_as_chunks)
        
        for page in group_with_vectors:
            doc_id = f"{pdf_name}_page_{page['page']}"
            documents.append({
                "_id": doc_id,
                "pdf_name": pdf_name,
                "text": page["text"],
                "page": page["page"],
                "patient_name": metadata["patient_name"],
                "patient_id": metadata["patient_id"],
                "final_bill": metadata["final_bill"],
                "disease": metadata["disease"],
                "$vector": page["vector"]
            })
            
    print(f"Uploading {len(documents)} pages to Astra DB...")
    batch_size = 50
    inserted_count = 0
    
    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        res = collection.insert_many(batch)
        inserted_count += len(res.inserted_ids)
        
    print(f"Successfully ingested {inserted_count} pages into collection '{args.collection}'.")


def query_command(args):
    """Query similarity search on Astra DB using embedded query text."""
    db = get_db()
    
    # Load model and vectorize query
    print("Loading SentenceTransformer model to encode query...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = model.encode(args.query).tolist()
    
    # Check collection
    collections = db.list_collection_names()
    if args.collection not in collections:
        print(f"Error: Collection '{args.collection}' does not exist in the database.", file=sys.stderr)
        sys.exit(1)
        
    collection = db.get_collection(args.collection)
    
    print(f"Querying collection '{args.collection}' for similarity to: '{args.query}'")
    results = collection.find(
        {},
        sort={"$vector": query_vector},
        limit=args.limit,
        include_similarity=True
    )
    
    print("\n--- Search Results ---")
    results_list = list(results)
    if not results_list:
        print("No matching documents found.")
        return
        
    for idx, doc in enumerate(results_list):
        similarity = doc.get("$similarity", 0.0)
        pdf_info = f"[PDF: {doc.get('pdf_name')}, Page {doc.get('page')}"
        if doc.get("chunk_index") is not None and doc.get("chunk_index") != 0:
            pdf_info += f", Chunk {doc.get('chunk_index')}"
        pdf_info += "]"
        
        print(f"\nResult #{idx+1} [Similarity: {similarity:.4f}] {pdf_info}")
        if any(k in doc for k in ["patient_name", "patient_id", "disease", "final_bill"]):
            print(f"Metadata:")
            print(f"  Patient Name : {doc.get('patient_name', 'N/A')}")
            print(f"  Patient ID   : {doc.get('patient_id', 'N/A')}")
            print(f"  Disease      : {doc.get('disease', 'N/A')}")
            print(f"  Final Bill   : {doc.get('final_bill', 'N/A')}")
        print(f"Text Content:\n{doc.get('text')}")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="PDF Ingestion and Vector Storage tool for DataStax Astra DB.")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to execute")
    
    # Ingest subcommand parser
    ingest_parser = subparsers.add_parser("ingest", help="Parse, chunk, vectorize, and ingest a PDF file")
    ingest_parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    ingest_parser.add_argument("--collection", default="pdf_chunks", help="Astra DB collection name")
    ingest_parser.add_argument("--chunk-size", type=int, default=500, help="Text chunk size in characters")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=50, help="Text chunk overlap in characters")
    
    # Query subcommand parser
    query_parser = subparsers.add_parser("query", help="Search the database for similarity matching")
    query_parser.add_argument("--query", required=True, help="The search query text")
    query_parser.add_argument("--collection", default="pdf_chunks", help="Astra DB collection name")
    query_parser.add_argument("--limit", type=int, default=3, help="Maximum number of results to display")
    
    # Print help and exit if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
        
    args = parser.parse_args()
    
    if args.command == "ingest":
        ingest_command(args)
    elif args.command == "query":
        query_command(args)


if __name__ == "__main__":
    main()
