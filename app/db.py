#from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError
from pathlib import Path
from functools import lru_cache
import json

MONGO_URL = "mongodb://mongo:27017"  # 'mongo' is service name in docker-compose
DB_NAME = "metadata_db"
COLLECTION_NAME = "metadata"

@lru_cache
def get_client() -> MongoClient:
    """Creates and returns the Synchronous PyMongo client."""
    print(f"Connecting to synchronous MongoDB at {MONGO_URL}...")
    # This call is blocking, but it's only called once at startup thanks to @lru_cache
    return MongoClient(MONGO_URL)

def get_db_connection() -> Collection:
    """Returns the Collection object without performing any insertion."""
    client = get_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return collection
# def get_client() -> AsyncIOMotorClient:
#     return AsyncIOMotorClient(MONGO_URL)

def get_collection():

    client = get_client()
    
    # Access the database and collection
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    print(f"Connected to DB: '{DB_NAME}', Collection: '{COLLECTION_NAME}'")

    is_empty = collection.count_documents({}) == 0

    if is_empty:

        try:
            with open('s3_crop_meta.json', 'r') as f:
                METADATA_ARRAY = json.load(f)
            print(f"Successfully loaded {len(METADATA_ARRAY)} documents from example_data.json.")
            return insert_metadata_objects(METADATA_ARRAY, collection)
    
        except FileNotFoundError:
            print("Error: s3_crop_meta.json not found. ")
        
        except json.JSONDecodeError:
            print("Error: s3_crop_meta.json contains invalid JSON. ")
    
    return collection




def insert_metadata_objects(data_array, collection):
    """
    Connects to MongoDB using the asynchronous 'motor' driver and performs a bulk insert 
    using insert_many() with asyncio.
    """
    if not data_array:
        print("The data array is empty. Nothing to insert.")
        return

    try:

        result = collection.insert_many(data_array)
        
        print("\n--- Insertion Successful ---")
        print(f"Total documents inserted: {len(result.inserted_ids)}")

        return collection
        
    except BulkWriteError as bwe:
        # This error is raised if some, but not all, documents failed insertion
        print("\n--- Insertion Failed with Partial Success ---")
        print(f"Documents successfully inserted: {bwe.details.get('nInserted')}")
        print("Write Errors:", bwe.details.get('writeErrors'))
        print("Full error details:", bwe.details)
        
    except Exception as e:
        # Handle general connection or parsing errors
        print(f"\n--- Critical Error During Preload ---")
        print(f"An unexpected error occurred: {e}")
        
    # finally:
    #     # Close the connection whether insertion succeeded or failed
    #     if client:
    #         client.close()
    #         print("MongoDB connection closed.")
