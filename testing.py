from ddgs import DDGS
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from newspaper import Article
import nltk
import csv
import os
import time

# Télécharger le modèle de tokenisation
nltk.download('punkt')

query = "actualités trading bourse"
output_file = "articles_results.csv"

query_list = ["actualités trading bourse", "trading articles", "elon musk tesla"]
news_sources = ["boursorama.com", "lesechos.fr", "investing.com", "reuters.com", "zonebourse.com", "lefigaro.fr", "france24.com"]


final_query_list = []
for q in query_list:
    for source in news_sources:
        final_query_list.append(q + " site:" + source)

for q in final_query_list:
    print(q)
        
# Création du CSV avec en-tête s'il n'existe pas
file_exists = os.path.isfile(output_file)
with open(output_file, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Titre", "URL", "Résumé", "Résumé auto", "Auteurs", "Date", "Mots-clés"])


with DDGS() as ddgs:
    results = list(ddgs.news(query, region="fr-fr", max_results=10, safesearch="off", timelimit="w"))
    print(f"Nombre de résultats trouvés : {len(results)}")

    for result in results:
        url = result.get("url")
        titre = result.get("title", "(titre manquant)")
        source = result.get("source", "")
        print(f"\n{titre}\n{url}")

        if not url:
            continue

        # Filtrage des sites
        if not any(domain in url.lower() for domain in news_sources):
            print(f"Source non légitime : {source or url}")
            continue

        try:
            # Téléchargement et parsing
            article = Article(url, language="fr")
            article.download()
            article.parse()
            article.nlp()

            # Récupération des infos
            titre = article.title or titre
            resume = article.summary.replace("\n", " ") if article.summary else "(Résumé non disponible)"
            auteurs = ", ".join(article.authors) if article.authors else "(inconnu)"
            date = article.publish_date if article.publish_date else "(inconnue)"
            motscles = ", ".join(article.keywords) if article.keywords else "(aucun)"

            # Résumé automatique avec Sumy
            if article.text.strip():
                parser = PlaintextParser.from_string(article.text, Tokenizer("french"))
                summarizer = LexRankSummarizer()
                summary_sentences = summarizer(parser.document, 3)
                resume_auto = " ".join(str(sentence) for sentence in summary_sentences)
            else:
                resume_auto = "(pas de texte)"
            
            # Filtrage léger des articles vides
            if len(article.text.split()) < 50:
                print("Article trop court, ignoré.")
                continue

            print(f"Ajouté : {titre}")

            # Sauvegarde dans le CSV
            with open(output_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([titre, url, resume, resume_auto, auteurs, date, motscles])

        except Exception as e:
            print(f"Erreur avec {url}: {type(e).__name__} - {e}")

        time.sleep(2)

print(f"\nExtraction terminée, résultats enregistrés dans {output_file}")

"""
- chaque heure, fetch une grande liste d'article

- pour fetcher les urls, on utiliser DDGs, mais aussi les classes news_scrapper
- checker si l'article a deja ete extrait, avant de refaire l'extraction, pour ne pas re-extracter plusieurs fois

    - pour chaque article fetcher, on vas determiner si l'article est important ou pas par rapport au theme 
    
    - pour savoir pour si l'article est important, on demande a un llm:
        -  "le theme du system est Trading News, determine si cet article merite d'etre sauvegardé ou pas. Si il doit etre sauvegardé, return True, sinon return False "
        - si oui, sauvegarder l'article dans la bd, si non, ignorer l'article
        
    - pour chaque article importants, on va l'enregistrer dans une base de données
    - chaque object d'article dans la bd doit avoir:
        - titre
        - url
        - resume
        - resume_auto
        - auteurs
        - date
        - motscles
    
    - a chaque que le pipeline est lancé, on va aussi stocker dans la bd des logs par rapport au pipeline
    
Base de données:
- MongoDB
 - collection: articles
    - document:
        - titre
        - url
        - resume
        - resume_auto
        - auteurs
        
- collection: pipeline_log
    - document:
        - heure à laquelle on fetch les articles
        - le nombre d'url fetcher
        - le nombre article extrait
        - le nombre d'article importants et enregistrer

- Pour plus tard, il faudra prevoir comment tout deployer sur le cloud (soit GCP "google cloud run")

Cron Job:
- run the pipeline every {period}


"""