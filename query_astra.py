"""
Query Astra Vector Database
Search through ingested PDF documents using natural language queries.
"""

import os
from dotenv import load_dotenv
from astrapy import DataAPIClient
import argparse

# Load environment variables
load_dotenv()

# Get Astra DB credentials from environment
ASTRA_DB_API_ENDPOINT = os.getenv("API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("APPLICATION_TOKEN")

def query_astra(query_text, collection_name="pdf_documents", limit=5):
    """Query the Astra DB vector database."""
    
    # Verify environment variables
    if not ASTRA_DB_API_ENDPOINT or not ASTRA_DB_APPLICATION_TOKEN:
        raise ValueError("Missing Astra DB credentials. Check your .env file.")
    
    print(f"\n[*] Connecting to Astra DB...")
    client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
    database = client.get_database_by_api_endpoint(ASTRA_DB_API_ENDPOINT)
    
    print(f"[*] Accessing collection '{collection_name}'...")
    collection = database.get_collection(collection_name)
    
    print(f"[*] Searching for: '{query_text}'")
    print(f"[*] Retrieving top {limit} results...\n")
    
    # Perform vector similarity search using $vectorize
    results = collection.find(
        sort={"$vectorize": query_text},
        limit=limit,
        projection={"*": True},
        include_similarity=True
    )
    
    # Display results
    print("=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)
    
    result_count = 0
    for i, doc in enumerate(results, 1):
        result_count = i
        print(f"\n[Result {i}]")
        print(f"Document ID: {doc.get('_id', 'N/A')}")
        print(f"Source: {doc.get('source', 'N/A')}")
        print(f"Chunk: {doc.get('chunk_id', 'N/A')} of {doc.get('total_chunks', 'N/A')}")
        
        # Show similarity score if available
        if '$similarity' in doc:
            similarity_score = doc['$similarity']
            print(f"Similarity Score: {similarity_score:.4f}")
        
        # Show the vectorized text content
        vectorized_text = doc.get('$vectorize', 'No text content')
        print(f"\nContent Preview:")
        print("-" * 80)
        # Show first 500 characters, handling Unicode errors
        preview = vectorized_text[:500] if len(vectorized_text) > 500 else vectorized_text
        # Encode and decode to handle special characters
        try:
            print(preview.encode('ascii', 'ignore').decode('ascii'))
        except:
            print(preview.encode('utf-8', 'ignore').decode('utf-8', 'ignore'))
        if len(vectorized_text) > 500:
            print(f"\n... (showing {len(preview)} of {len(vectorized_text)} characters)")
        print("-" * 80)
    
    if result_count == 0:
        print("\n[!] No results found. The collection might be empty or the query didn't match any documents.")
    else:
        print(f"\n[+] Found {result_count} result(s)")
    
    print("=" * 80)

def list_collections():
    """List all collections in the database."""
    print(f"\n[*] Connecting to Astra DB...")
    client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
    database = client.get_database_by_api_endpoint(ASTRA_DB_API_ENDPOINT)
    
    print(f"[*] Fetching collections...")
    collections = database.list_collection_names()
    
    print("\n" + "=" * 60)
    print("AVAILABLE COLLECTIONS")
    print("=" * 60)
    for i, coll_name in enumerate(collections, 1):
        print(f"{i}. {coll_name}")
    print("=" * 60)
    print(f"\nTotal collections: {len(collections)}")

def main():
    parser = argparse.ArgumentParser(
        description="Query Astra Vector Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search in default collection
  python query_astra.py "What is biological aging?"
  
  # Search in specific collection with more results
  python query_astra.py "frailty index" --collection bioage_research --limit 10
  
  # List all collections
  python query_astra.py --list-collections
        """
    )
    
    parser.add_argument("query", nargs="?", help="Search query text")
    parser.add_argument("--collection", default="pdf_documents", help="Collection name to search")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    parser.add_argument("--list-collections", action="store_true", help="List all available collections")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ASTRA DB VECTOR SEARCH")
    print("=" * 80)
    
    try:
        if args.list_collections:
            list_collections()
        elif args.query:
            query_astra(args.query, args.collection, args.limit)
        else:
            print("\n[!] Error: Please provide a query or use --list-collections")
            parser.print_help()
    
    except Exception as e:
        print(f"\n[-] Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

