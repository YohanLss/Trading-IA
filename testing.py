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

# Création du CSV avec en-tête s'il n'existe pas
file_exists = os.path.isfile(output_file)
with open(output_file, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Titre", "URL", "Résumé", "Résumé auto", "Auteurs", "Date", "Mots-clés"])

news_sources = ["boursorama.com", "lesechos.fr", "investing.com", "reuters.com", "zonebourse.com", "lefigaro.fr", "france24.com"]

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
