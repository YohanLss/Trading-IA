from newsapi import NewsApiClient
from services.scrapers.base_news_scraper import BaseScraper, Article

class NewsApi(BaseScraper):
    
   
    def __init__(self, api_key=None, **kwargs):
        if api_key is None:
            raise ValueError("Missing api_key")
        self.newsapi = NewsApiClient(api_key=api_key)


    def get_everything(self,
        search,
        language="en",
        sources=None,
        domains=None,
        from_date=None,
        to_date=None,
        sort_by="publishedAt"):


        params = {
            "q": search,
            "language": language,
            "sort_by": sort_by
        }

        if sources is not None:
            params["sources"] = sources

        if domains is not None:
            params["domains"] = domains

        if from_date is not None:
            params["from_param"] = from_date

        if to_date is not None:
            params["to"] = to_date

        expected = self.newsapi.get_everything(**params)
        raw_articles = expected.get("articles", [])

        cleaned_articles = []

        for article in raw_articles:
            clean = {}
            source_block = article.get("source") or {}
            clean["source"] = source_block.get("name") or ""
            clean["author"] = article.get("author") or ""
            clean["title"] = article.get("title") or ""
            clean["description"] = article.get("description") or ""
            clean["content"] = article.get("content") or ""
            clean["url"] = article.get("url") or ""
            clean["publishedAt"] = article.get("publishedAt") or ""

            cleaned_articles.append(clean)

        return cleaned_articles


def main():
    
    API_KEY = "..."

    # Instancier le scraper
    news = NewsApi(api_key=API_KEY)

    # Test rapide : chercher des news sur "bitcoin"
    articles = news.get_everything("bitcoin")




    # Affichage propre
    print(f"Nombre d'articles trouvés : {len(articles)}\n")

    for i, article in enumerate(articles[:5]):  # affiche les 5 premiers
        print(f"=== ARTICLE {i+1} ===")
        print("Source      :", article["source"])
        print("Titre       :", article["title"])
        print("Description :", article["description"])
        print("URL         :", article["url"])
        print("Publié le   :", article["publishedAt"])
        print("---------------\n")

if __name__ == "__main__":
    main()

    

        




