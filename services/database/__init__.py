import os
from dotenv import load_dotenv
import certifi
load_dotenv()

from .article_service import ArticleService
from .pipeline_execution_service import PipelineExecutionService
from pymongo import MongoClient
from .untracked_symbols_service import UntrackedSymbolsService
uri = os.getenv("MONGO_URI")
db_client = MongoClient(uri, tlsCAFile=certifi.where())
db = db_client["dev"]

article_service = ArticleService(db)
pipeline_execution_service = PipelineExecutionService(db)
untracked_symbols_service = UntrackedSymbolsService(db)