from duckduckgo_search import DDGS
from googlesearch import search
from newspaper import Article
import nltk
import urllib
import csv
nltk.download('punkt')
nltk.download('punkt_tab')


query = "trading articles"
output_file = f"articles_results.csv"


with DDGS() as ddgs:
    results = ddgs.text(query, region="fr-fr", max_results=1)

    #print the 5 first URl 
    for result in results:
        url = result["href"]
        print(f"\n {url}")

        try:
            article = Article(url, language="fr")
            article.download()
            article.parse()
            article.nlp()

            titre = article.title
            resume = article.summary.replace("\n", " ")

            print(f" {titre}")
            print(f" {resume[:200]}...")
            print(article.authors)        # Liste des auteurs
            print(article.text)
            print(article.publish_date)   # Date de publication
            print(article.keywords)       # Mots-clés extraits automatiquement


            #results saved in a csv file
            with open(output_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([titre, url, resume])

        except Exception as e:
            print(f" Erreur avec {url}: {e}")

    print(f"\n Extraction terminée, résultats enregistrés dans {output_file}")
    
