import pymongo
from pymongo import MongoClient

uri = "mongodb+srv://local_dev:local_password@tradingia.y8wuale.mongodb.net/?retryWrites=true&w=majority&appName=TradingIA"

client = MongoClient(uri)

def insert_article(articles_collection, article):
    articles_collection.insert_one(article)


# def get_articles(articles_collection):
#     print(articles_collection.find())
try:
    database = client.get_database("test_db")
    articles_collection = database.get_collection("articles")
    # Query for a movie that has the title 'Back to the Future'
    query = { "title": "Back to the Future" }
    article = {
        "title": "Apple releases iphone",
        "publish_date": "2021-01-01",
        "url": "https://www.apple.com/news/iphone-13-pro-release-date/"   ,
        "summary": "New apple iphone."
    }
    # insert_article(articles_collection, article)
    
    # Find() == SELECT * 
    filter = {
        "title":"Apple"
    }
    fetched_articles = list(articles_collection.find(filter))
    
    articles_collection.delete_many(filter)
    for article in fetched_articles:
        print(article)
        
    client.close()
except Exception as e:
    raise Exception("Unable to find the document due to the following error: ", e)


# CRUD

# CREATE
# READ
# UPDATE
# DELETE
