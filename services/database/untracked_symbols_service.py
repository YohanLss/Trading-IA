import os

from bson import ObjectId
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timezone
from typing import Dict, List

from pymongo import MongoClient, UpdateOne, DESCENDING, ASCENDING
from models import Article
from services.llm import gemini_client
from utils import function_timer

class UntrackedSymbolsService:
    def __init__(self, db = None):
        if db is None:
            uri = os.getenv("MONGO_URI")
            db_client = MongoClient(uri)
            db = db_client["dev"]
        self.db = db
        
        self.collection = db.untracked_symbols
        
    def get_untracked_symbols(self) -> List[str]:
        untracked_symbols = list(self.collection.find({}, {"_id": 0, "symbol": 1}))
        return [symbol.get("symbol") for symbol in untracked_symbols]
    
    def add_untracked_symbols(self, symbols):
        now = datetime.now(timezone.utc)
        operations = []
        
        for symbol in symbols:
            operations.append(
                UpdateOne(
                    {"symbol": symbol},
                    {"$set": {"created_at": now}},
                    upsert=True
                )
            )
        if not operations:
            return {"inserted_count": 0, "status": "success", "error": None}
        try:
            res = self.collection.bulk_write(operations)
            return {
                "inserted_count": res.upserted_count,
                "status": "success",
                "error": None,
            }
        except Exception as e:
            raise Exception(f"Error inserting untracked symbols into database: {e}")