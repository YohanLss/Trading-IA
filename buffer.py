import requests
from fake_useragent import UserAgent
from IPython.display import HTML


ua = UserAgent()
headers = {'User-Agent': ua.random}

print(ua.random)
useragent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"

headers = {'User-Agent': ua.random}
url = "https://seekingalpha.com/api/v3/news?fields[news]=title%2Cdate%2Ccomment_count%2Ccontent%2CprimaryTickers%2CsecondaryTickers%2Ctag%2CgettyImageUrl%2CpublishOn&fields[tag]=slug%2Cname&filter[category]=market-news%3A%3Aall&filter[since]=0&filter[until]=0&include=primaryTickers%2CsecondaryTickers&isMounting=true&page[size]=25&page[number]=1"

url = 'https://seekingalpha.com/api/v3/news/4518979-sa-asks-whats-the-best-energy-etf-right-now'
response = requests.get(url, headers=headers)
text = response.text
print(response.text)

data = response.json().get("data")
if data and isinstance(data, list):
    print("success")
    print("fetched: ", len(data), " articles")
    print(data)
else:
    print("failed")
