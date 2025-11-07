import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING, UpdateOne

load_dotenv()


class PipelineExecutionService:
    """
    Minimal service for db.pipeline_executions:
      - get one / many
      - insert one / many
    """

    def __init__(self, db=None):
        if db is None:
            uri = os.getenv("MONGO_URI")
            db_client = MongoClient(uri)
            db = db_client["dev"]

        self.collection = db.pipeline_executions
        # Basic indexes for common queries
        self.collection.create_index([("start_time", DESCENDING)], name="start_time_desc")
        self.collection.create_index([("status", ASCENDING)], name="status_asc")

    # --------- Reads ---------

    def get_execution(self, _id: Union[ObjectId, str]) -> Optional[Dict]:
        if isinstance(_id, str):
            _id = ObjectId(_id)
        return self.collection.find_one({"_id": _id})

    def get_executions(
        self,
        filter: Optional[Dict] = None,
        sorting: Optional[Dict[str, int]] = None,
        limit: int = 100,
    ) -> List[Dict]:
        filter = filter or {}
        sorting = sorting or {"start_time": DESCENDING, "_id": DESCENDING}
        sort_list = [(k, v) for k, v in sorting.items()]
        return list(self.collection.find(filter=filter, sort=sort_list, limit=limit))

    # --------- Writes ---------

    def insert_one_execution(self, exec_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects a ready-to-insert dict. Adds _id and created_at if missing.
        """
        doc = dict(exec_doc)
        doc.setdefault("_id", ObjectId())
        doc.setdefault("created_at", datetime.now(timezone.utc))
        # normalize common fields if present
        if isinstance(doc.get("start_time"), tuple):
            doc["start_time"] = doc["start_time"][0]
        if isinstance(doc.get("end_time"), tuple):
            doc["end_time"] = doc["end_time"][0]

        try:
            res = self.collection.insert_one(doc)
            return {"inserted_id": str(res.inserted_id), "status": "success", "error": None}
        except Exception as e:
            return {"inserted_id": None, "status": "error", "error": str(e)}

    def insert_many_executions(self, exec_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Bulk insert. Adds _id and created_at where missing.
        """
        docs: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc)

        for raw in exec_docs:
            d = dict(raw)
            d.setdefault("_id", ObjectId())
            d.setdefault("created_at", now)
            if isinstance(d.get("start_time"), tuple):
                d["start_time"] = d["start_time"][0]
            if isinstance(d.get("end_time"), tuple):
                d["end_time"] = d["end_time"][0]
            docs.append(d)

        try:
            res = self.collection.insert_many(docs, ordered=False)
            return {
                "inserted_count": len(res.inserted_ids),
                "inserted_ids": [str(i) for i in res.inserted_ids],
                "status": "success",
                "error": None,
            }
        except Exception as e:
            return {"inserted_count": 0, "inserted_ids": [], "status": "error", "error": str(e)}
