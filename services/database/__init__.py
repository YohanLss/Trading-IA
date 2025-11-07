import os
from dotenv import load_dotenv
load_dotenv()

from .article_service import ArticleService
from .pipeline_execution_service import PipelineExecutionService
from pymongo import MongoClient

uri = os.getenv("MONGO_URI")
db_client = MongoClient(uri)
db = db_client["dev"]

article_service = ArticleService(db)
pipeline_execution_service = PipelineExecutionService(db)