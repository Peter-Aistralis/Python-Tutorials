import asyncio
import nest_asyncio #only needed when you're using Spyder
import json
from agents import Agent, function_tool, Runner
import textwrap

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
        "symbol": symbol
    }
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": CMC_KEY
    }
    response2 = requests.get(url, headers=headers, params=params)
    data = response2.json()
    return data["data"][symbol][0]["quote"]["USD"]["price"]

@function_tool
def send_mail(name: str, message: str) -> str:
    """Send a mail to someone with a message"""

    return F"email has been sent to {name}"


agent = Agent(
    name="Riri",
    instructions='you are a cute helpful asistant called Riri and you respond in a very funny way',
    tools=[get_price, send_mail],
    model="gpt-4o-mini"
)


async def main():
    result = await Runner.run(agent, "What is the price of bitcoin and Cardano. send both price by mail to Peter")
    print (result)
    raw_dicts = [vars(resp) for resp in result.raw_responses]
    print(json.dumps(raw_dicts, indent=2, default=str))
    # Calculate total tokens across all turns
    total_tokens = sum(
        resp.usage.total_tokens 
        for resp in result.raw_responses 
        if hasattr(resp, "usage") and resp.usage
    )
    print(f"🔥 Total Tokens Spent: {total_tokens}")
    print(textwrap.fill(result.final_output, width=80))
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
    #result = Runner.run_sync(agent, "What is the price of bitcoin?")
    #print(result.final_output)
#await main()