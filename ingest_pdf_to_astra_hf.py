"""
PDF to Astra Vector Database Ingestion Script (HuggingFace Embeddings)
This script uses FREE HuggingFace embeddings instead of OpenAI.
"""

import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import AstraDB
from langchain_community.embeddings import HuggingFaceEmbeddings
import argparse

# Load environment variables
load_dotenv()

# Get Astra DB credentials from environment
ASTRA_DB_API_ENDPOINT = os.getenv("API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("APPLICATION_TOKEN")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    print(f"[*] Reading PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    
    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        text += page_text + "\n\n"
        print(f"   Processed page {page_num}/{len(reader.pages)}")
    
    print(f"[+] Extracted {len(text)} characters from {len(reader.pages)} pages")
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks for embedding."""
    print(f"\n[*] Chunking text (size={chunk_size}, overlap={chunk_overlap})")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    print(f"[+] Created {len(chunks)} chunks")
    
    return chunks

def ingest_to_astra(chunks, collection_name="pdf_documents", pdf_filename=""):
    """Ingest text chunks into Astra DB vector database."""
    print(f"\n[*] Ingesting to Astra DB collection: '{collection_name}'")
    
    # Verify environment variables
    if not ASTRA_DB_API_ENDPOINT or not ASTRA_DB_APPLICATION_TOKEN:
        raise ValueError("Missing Astra DB credentials. Check your .env file.")
    
    # Initialize HuggingFace embeddings (FREE - no API key needed)
    print(f"   Loading HuggingFace embeddings model (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"   Embeddings model loaded successfully!")
    
    # Create documents with metadata
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "page_content": chunk,
            "metadata": {
                "source": pdf_filename,
                "chunk_id": i,
                "total_chunks": len(chunks)
            }
        })
    
    # Initialize Astra DB vector store
    print(f"   Connecting to Astra DB...")
    vstore = AstraDB(
        embedding=embeddings,
        collection_name=collection_name,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
    )
    
    # Add documents to vector store
    print(f"   Adding {len(documents)} documents...")
    texts = [doc["page_content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    
    vstore.add_texts(texts=texts, metadatas=metadatas)
    
    print(f"[+] Successfully ingested {len(documents)} chunks to Astra DB!")
    print(f"   Collection: {collection_name}")
    print(f"   Source: {pdf_filename}")
    
    return vstore

def main():
    parser = argparse.ArgumentParser(description="Ingest PDF into Astra Vector Database (HuggingFace)")
    parser.add_argument("pdf_path", help="Path to the PDF file to ingest")
    parser.add_argument("--collection", default="pdf_documents", help="Astra DB collection name")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    
    args = parser.parse_args()
    
    # Verify PDF file exists
    if not os.path.exists(args.pdf_path):
        print(f"[-] Error: PDF file not found: {args.pdf_path}")
        return
    
    pdf_filename = os.path.basename(args.pdf_path)
    
    print("=" * 60)
    print("PDF TO ASTRA VECTOR DATABASE INGESTION")
    print("Using HuggingFace Embeddings (FREE)")
    print("=" * 60)
    print(f"PDF File: {pdf_filename}")
    print(f"Collection: {args.collection}")
    print(f"Chunk Size: {args.chunk_size}")
    print(f"Chunk Overlap: {args.chunk_overlap}")
    print("=" * 60)
    print()
    
    try:
        # Step 1: Extract text from PDF
        text = extract_text_from_pdf(args.pdf_path)
        
        if not text.strip():
            print("[-] Error: No text extracted from PDF. The PDF might be image-based.")
            return
        
        # Step 2: Chunk the text
        chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
        
        # Step 3: Ingest to Astra DB
        vstore = ingest_to_astra(chunks, args.collection, pdf_filename)
        
        print("\n" + "=" * 60)
        print("[+] INGESTION COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[-] Error during ingestion: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



