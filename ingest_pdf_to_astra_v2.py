"""
PDF to Astra Vector Database Ingestion Script (Updated for newer astrapy)
This script uses FREE HuggingFace embeddings.
"""

import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from astrapy import DataAPIClient
import argparse
import time

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
    print(f"   Loading HuggingFace embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"   Embeddings model loaded successfully!")
    
    # Get embedding dimension
    print(f"   Testing embeddings...")
    sample_embedding = embeddings.embed_query("test")
    embedding_dim = len(sample_embedding)
    print(f"   Embedding dimension: {embedding_dim}")
    
    # Initialize Astra DB client
    print(f"   Connecting to Astra DB...")
    client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
    database = client.get_database(ASTRA_DB_API_ENDPOINT)
    
    # Create or get collection
    print(f"   Creating/getting collection '{collection_name}'...")
    try:
        collection = database.create_collection(collection_name)
        print(f"   Collection created successfully!")
    except Exception as e:
        print(f"   Collection may already exist, attempting to get it...")
        collection = database.get_collection(collection_name)
        print(f"   Using existing collection!")
    
    # Prepare documents
    print(f"   Preparing {len(chunks)} documents...")
    documents = []
    
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"   Processing chunk {i+1}/{len(chunks)}...")
        
        # Generate embedding
        embedding = embeddings.embed_query(chunk)
        
        # Create document
        doc = {
            "_id": f"{pdf_filename}_chunk_{i}",
            "text": chunk,
            "$vector": embedding,
            "metadata": {
                "source": pdf_filename,
                "chunk_id": i,
                "total_chunks": len(chunks)
            }
        }
        documents.append(doc)
    
    # Insert documents in batches
    print(f"   Inserting documents into Astra DB...")
    batch_size = 20
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        collection.insert_many(batch)
        print(f"   Inserted batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        time.sleep(0.1)  # Small delay to avoid rate limits
    
    print(f"[+] Successfully ingested {len(documents)} chunks to Astra DB!")
    print(f"   Collection: {collection_name}")
    print(f"   Source: {pdf_filename}")
    
    return collection

def main():
    parser = argparse.ArgumentParser(description="Ingest PDF into Astra Vector Database")
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
        collection = ingest_to_astra(chunks, args.collection, pdf_filename)
        
        print("\n" + "=" * 60)
        print("[+] INGESTION COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[-] Error during ingestion: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

