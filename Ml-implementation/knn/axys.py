import yfinance as yf

# Télécharger les données
data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

# Sauvegarder en CSV
data.to_csv("AAPL.csv")
