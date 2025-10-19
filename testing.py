from duckduckgo_search import DDGS
from googlesearch import search
from newspaper import Article
import nltk
import urllib
import csv
import os
nltk.download('punkt')
nltk.download('punkt_tab')


query = "trading articles"
output_file = f"articles_results.csv"

# Création du CSV avec en-tête
file_exists = os.path.isfile(output_file)
with open(output_file, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Titre", "URL", "Résumé", "Auteurs", "Date", "Mots-clés"])


#liste les sites legit 
news_sources = ["boursorama.com", "lesechos.fr", "investing.com", "reuters.com", "zonebourse.com"]

#renvoie un dictionnaire avec clé
with DDGS() as ddgs:
    results = ddgs.news(query, region="fr-fr", max_results=5, safesearch="off", timelimit="w")

    #print the 5 first URl 
    for result in results:
        url = result["url"]
        print(f"\n {url}")

        #filtre parmi les sites resultants de la recherche
        if not any(domain in url for domain in news_sources):
            continue

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
            print(f"Erreur avec {url}: {type(e).__name__} - {e}")  #concernant les erreurs de reseaux ou parsing

    print(f"\n Extraction terminée, résultats enregistrés dans {output_file}")
    
