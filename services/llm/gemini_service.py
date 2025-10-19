from google import genai
from google.genai.types import GenerateContentConfig, CreateBatchJobConfig

from utils import logger
import os
import json
from typing import List, Optional
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from models import Article


load_dotenv()

key = os.getenv("GEMINI_API_KEY")
class LlmSummary(BaseModel):
    """
    Represents the expected JSON structure returned by the LLM summarization request.
    Contains one optional 'summary' field.
    """
    summary: Optional[str] = None


class GeminiService:
    """
    Wraps around the Google Gemini API client and provides helper methods to initialize the client,
    send requests, and summarize financial news articles.
    """
    def __init__(self, api_key):
        """
        Initializes the Gemini API client using the provided API key.
        """
        self.client = genai.Client(api_key=api_key)
        if not self.client_is_initalized():
            self.client = None

    def client_is_initalized(self):
        """
        Tests whether the Gemini client can successfully communicate with the API.
        Returns True if initialization is successful, False otherwise.
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents="Hello"
            )
            if response.text:
                logger.info("Gemini API initialized successfully.")
            else:
                logger.info("Gemini API is reachable, but the response was empty.")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini API: {e}")
            return False

    def send_request(self, prompt_data, sys_instruct: str = "", schema=None):
        """
        Sends a prompt with optional system instructions and schema to the Gemini API.
        Returns the parsed response if a schema is provided, otherwise returns the raw response text.
        """
        try:
            payload = json.dumps(prompt_data, ensure_ascii=False, indent=2)
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=[payload],
                config=GenerateContentConfig(
                    system_instruction=sys_instruct,
                    response_mime_type="application/json",
                    response_schema=schema
                )
            )
            res = response.parsed if schema else response.text
            return res
        except Exception as e:
            logger.error(f"Error in sending request to Gemini API: {e}")
            return None

    def batch_summarize_articles(self, articles: List[Article]):
        """
        Processes a batch of articles to summarize them.
        (Currently not implemented.)
        """
        requests = []
        for article in articles:
            pass

    def summarize_article(self, article: Article):
        """
        Sends a summarization request for a single Article and returns a one-sentence summary.
        """
        article_dict = article.model_dump()
        article_dict["summary"] = ""
        sys_intruct = "You are a precise financial news assistant."
        prompt_data = {
            "Task": "Use one sentence to summarize the given article",
            "Article": article_dict,
        }
        payload = json.dumps(prompt_data, ensure_ascii=False, indent=2)
        schema = LlmSummary

        response = self.send_request(prompt_data=payload, sys_instruct=sys_intruct, schema=schema)
        summary = response.summary

        return summary

    def send_text_request(self, message: str):
        """
        Sends a simple text message to the Gemini API and returns the response.
        """
        if not isinstance(message, str):
            raise TypeError("Message must be a string.")

        res = self.send_request(prompt_data=message)
        return res

def main():
    # yahoo_scraper = YahooScraper(limit=5, async_scrape=True)
    # print(yahoo_scraper.scrape())

    fetched_articles = [Article(url='https://finance.yahoo.com/news/protesters-oppose-trump-policies-no-163411318.html',
                                title='Protesters Oppose Trump Policies in ‘No Kings’ Events Across US',
                                content='(Bloomberg) --Demonstrators across the US turned out for what organizers said would be as many as 2,700 “No Kings” protests in all 50 states to express their opposition to President Donald Trump’s agenda. Saturday’s mass protests follow similar “No Kings” protests on June 14, timed to offset the military parade Trump hosted the same day in Washington for the 250th anniversary of the US Army and his birthday. Organizers estimated that 4 million to 6 million people attended the June demonstrations. Most Read from Bloomberg Affordable Housing Left Vulnerable After Trump Fires Building Inspectors Los Angeles County Declares State of Emergency Over ICE Raids What Comes After the\xa0‘War on Cars’? NY Senator Behind Casino Push Urges Swift Awarding of Licenses Trump Floats San Francisco as Next Target for Crime Crackdown Protests are also planned in Western Europe. The US government has been shut down for 18 days as Senate Democrats and Republicans remain dug in over extending health care subsidies, a roadblock to a spending bill that would reopen the government. The protesters are trying to show public opposition to Trump’s push to send National Guard troops to US cities, his immigration raids and his cuts to foreign aid and domestic programs favored by Democrats. Most Read from Bloomberg Businessweek Inside the Credit Card Battle to Win America’s Richest Shoppers Robinhood Is Banking on Babies and 401(k)s to Get Everyone Trading NBA Commissioner Adam Silver Has a Steve Ballmer Problem on His Hands The Banker Behind the Trumps’ Quick Wall Street Wins Meet Polymarket’s $400 Million Man ©2025 Bloomberg L.P.',
                                publish_date='2025-10-18 16:34:11+00:00', authors=['María Paula Mijares Torres'],
                                summary='Demonstrators across the US turned out for what organizers said would be as many as 2,700 “No Kings” protests in all 50 states to express their opposition to President Donald Trump’s agenda.\nSaturday’s mass protests follow similar “No Kings” protests on June 14, timed to offset the military parade Trump hosted the same day in Washington for the 250th anniversary of the US Army and his birthday.\nOrganizers estimated that 4 million to 6 million people attended the June demonstrations.\nMost Read from Bloomberg Affordable Housing Left Vulnerable After Trump Fires Building Inspectors Los Angeles County Declares State of Emergency Over ICE Raids What Comes After the ‘War on Cars’?\nNY Senator Behind Casino Push Urges Swift Awarding of Licenses Trump Floats San Francisco as Next Target for Crime Crackdown Protests are also planned in Western Europe.'),
                        Article(
                            url='https://finance.yahoo.com/news/brexit-hurt-economy-foreseeable-future-162559450.html',
                            title='Brexit will hurt economy for ‘foreseeable future’, claims Bailey',
                            content='Andrew Bailey has claimed Brexit will hurt the economy for years to come. The Governor of the Bank of England said the impact would be “negative” for the “foreseeable future” as he warned thatputting up trade barriersalways damaged growth. While Mr Bailey stressed that he was not offering a personal view of Brexit, he said that years of low UK productivity had driven up debt. He added: “What’s the impact on economic growth? As a public official, I have to answer that question – and the answer is that, for the foreseeable future, it’s negative.” Speaking in Washington DC, Mr Bailey said the economy was adjusting slowly to new trading relationships with “some partial rebalancing” in trade already taking place. In a thinly-veiled jibe at Donald Trump, Mr Bailey also warned against erecting global trade barriers. “If you make the world economy less open, it will have an effect on growth. It will reduce growth over time,” he said. Mr Bailey added: “Longer term, you’ll get some adjustment. Trade does adjust. It does rebuild, and all the evidence we have from the UK is that is exactly what is happening.” His remarks came as the International Monetary Fund admitted this week that the steep tariff increases imposed by the US president had been less damaging than previously feared. Meanwhile, Britain’s position outside the EU has enabled it to negotiate lower tariffs with the world’s biggest economy. The EU currently faces a 15pc levy on most goods exported to the US, compared with the UK’s 10pc rate. However, Mr Bailey also warned that years of low productivity had pushed up debt. He calculated that if growth over the past 15 years had matched the average rate seen before the financial crisis, Britain’s debt-to-GDP ratio would now be 82pc instead of 96pc and would be below 80pc by the end of the decade. “That is a big difference,” he warned. Rachel Reeves is expected to blame Brexit in the Budget for theOffice for Budget Responsibility’s (OBR)widely expected decision to lower its long-term growth forecasts. Economists have warned that her record £40bn tax raid has driven up prices and stifled growth, with the Chancellor expected to raise taxes by another £30bn in her second Budget on Nov 26 tobalance the books. Mr Bailey suggested that the lower “speed limit” of the economy made it harder for the Bank to keep rates low because the economy was now more vulnerable to inflation: “Slower growth makes economic policymaking it much more difficult.” He warned of the risks posed from the rapid rise of private credit issued by non-banks, adding that policymakers would do more to “lift the lid” on the sector. Mr Bailey also suggested that policymakers were eyeing reforms that would makethe gilt marketless susceptible to financial stability risks. Separate analysis backed by Lord Cameron, the former prime minister, warned that Britain was in danger of losing its rich country status. Prosperity Through Growth, a new book by authors including former Ronald Reagan adviser Art Laffer and Australian businessman Lord Hintze, showed the average Lithuanian would have a higher living standard than the average Briton by the end of the decade. It also warned the UK would drop from being the 25th richest country in the world 25 years ago to the 46th richest by 2050. Lord Cameron said: “We’re in a situation of genteel decline – people just putting up with 1pc growth. We’re effectively getting poorer, but we’re pretending we aren’t, and we need to convince people it doesn’t have to be that way. But we need to convince people we’ve got a very clear plan.” Broaden your horizons with award-winning British journalism. Try The Telegraph free for 1 month with unlimited access to our award-winning website, exclusive app, money-saving offers and more.',
                            publish_date='2025-10-18 16:25:59+00:00', authors=['Szu Ping Chan'],
                            summary='Andrew Bailey has claimed Brexit will hurt the economy for years to come.\nWhile Mr Bailey stressed that he was not offering a personal view of Brexit, he said that years of low UK productivity had driven up debt.\nIn a thinly-veiled jibe at Donald Trump, Mr Bailey also warned against erecting global trade barriers.\nMr Bailey added: “Longer term, you’ll get some adjustment.\nMr Bailey also suggested that policymakers were eyeing reforms that would make the gilt market less susceptible to financial stability risks.'),
                        ]
    # # print(len(fetched_articles))
    service = GeminiService(api_key=key)
    # # print(service.send_text_request("Hi what's up?"))
    # 
    print(service.summarize_article(fetched_articles[0]))
    # summary = service.summarize_article(fetched_articles[0])
    # print(summary)
if __name__ == "__main__":
    main()

