"""
Simple test to verify Astra DB connection
"""

import os
from dotenv import load_dotenv
from astrapy import DataAPIClient

load_dotenv()

ASTRA_DB_API_ENDPOINT = os.getenv("API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("APPLICATION_TOKEN")

print("=" * 60)
print("ASTRA DB CONNECTION TEST")
print("=" * 60)
print(f"Endpoint: {ASTRA_DB_API_ENDPOINT}")
print(f"Token: {ASTRA_DB_APPLICATION_TOKEN[:20]}...")
print("=" * 60)

try:
    print("\n[*] Creating client...")
    client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
    
    print("[*] Getting database...")
    database = client.get_database_by_api_endpoint(ASTRA_DB_API_ENDPOINT)
    
    print("[*] Listing collections...")
    collections = database.list_collection_names()
    
    print("\n[+] SUCCESS! Connection working.")
    print(f"\nFound {len(collections)} collection(s):")
    for coll in collections:
        print(f"  - {coll}")
    
except Exception as e:
    print(f"\n[-] ERROR: {e}")
    import traceback
    traceback.print_exc()


