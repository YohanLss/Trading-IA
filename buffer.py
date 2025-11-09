import requests
from fake_useragent import UserAgent
from IPython.display import HTML


ua = UserAgent()
headers = {'User-Agent': ua.random}


headers = {'User-Agent': ua.random}
url = "https://seekingalpha.com/api/v3/news?fields[news]=title%2Cdate%2Ccomment_count%2Ccontent%2CprimaryTickers%2CsecondaryTickers%2Ctag%2CgettyImageUrl%2CpublishOn&fields[tag]=slug%2Cname&filter[category]=market-news%3A%3Aall&filter[since]=0&filter[until]=0&include=primaryTickers%2CsecondaryTickers&isMounting=true&page[size]=25&page[number]=1"

url = 'https://seekingalpha.com/api/v3/news/4518979-sa-asks-whats-the-best-energy-etf-right-now'
response = requests.get(url, headers=headers)

