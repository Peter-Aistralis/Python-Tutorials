import asyncio
import nest_asyncio
import json
from agents import Agent, function_tool, Runner
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import requests

nest_asyncio.apply()


#OPENAI_API_KEY = os.getenv("Openai_Key")
CMC_KEY = os.getenv("CMC_Key")

#client = OpenAI(api_key=OPENAI_KEY)

@function_tool
def get_price(symbol: str) -> float:
    """Get the current price for a cryptocurrency."""
    url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest'
    params = {
        "symbol": symbol.upper()
    }
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": CMC_KEY
    }
    response2 = requests.get(url, headers=headers, params=params)
    data = response2.json()
    
    try:
        # CMC v2 returns a list under the symbol key
        coin_list = data["data"][symbol.upper()]
        return float(coin_list[0]["quote"]["USD"]["price"])
    except (KeyError, IndexError, TypeError) as e:
        print(f"⚠️ Error parsing CoinMarketCap response: {e}, Data: {data}")
        return 0.0


agent = Agent(
    name="Riri",
    instructions='you are a cute helpful asistant called Riri and you respond in a very funny way',
    tools=[get_price],
    model="gpt-4o-mini"
)


async def main():
    result = await Runner.run(agent, "What is the price of bitcoin?")
    print (result)
    history_json = json.dumps(result.to_input_list(), indent=2)
    print(history_json)
    # Convert each ModelResponse object's attributes into a dictionary
    raw_dicts = [vars(resp) for resp in result.raw_responses]
    print(json.dumps(raw_dicts, indent=2, default=str))
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
    #result = Runner.run_sync(agent, "What is the price of bitcoin?")
    #print(result.final_output)
#await main()