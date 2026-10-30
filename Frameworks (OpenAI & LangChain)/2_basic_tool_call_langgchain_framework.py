import asyncio
import nest_asyncio
import json
import os
import textwrap
import requests
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()
nest_asyncio.apply()

CMC_Key = os.getenv("CMC_Key")

# 1. Define tools using LangChain's @tool decorator
@tool
def get_price(symbol: str) -> float:
    """Get the current price for a cryptocurrency."""
    url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest'
    params = {
        "symbol": symbol
    }
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": CMC_Key
    }
    response2 = requests.get(url, headers=headers, params=params)
    data = response2.json()
    return data["data"][symbol][0]["quote"]["USD"]["price"]

@tool
def send_mail(name: str, message: str) -> str:
    """Send a mail to someone with a message"""
    return f"email has been sent to {name}"

tools = [get_price, send_mail]

# 2. Initialize the Chat Model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. Create the Agent using LangGraph (using 'prompt' instead of 'state_modifier')
system_prompt = "you are a cute helpful assistant called Riri and you respond in a very funny way"

agent_executor = create_agent(
    model, 
    tools, 
    system_prompt=system_prompt
)

async def main():
    inputs = {"messages": [("user", "What is the price of bitcoin and Cardano. send both price by mail to Peter")]}
    
    # Run the agent asynchronously
    response = await agent_executor.ainvoke(inputs)
    
    # Print the full message history as formatted JSON
    messages_dump = [msg.model_dump() for msg in response["messages"]]
    print(json.dumps(messages_dump, indent=2, default=str))
    
    total_tokens = 0
    for msg in response["messages"]:
        # Check if the message has response metadata containing token usage
        if hasattr(msg, "response_metadata") and "token_usage" in msg.response_metadata:
            total_tokens += msg.response_metadata["token_usage"].get("total_tokens", 0)
        # Some versions/models store it under usage_metadata directly
        elif hasattr(msg, "usage_metadata") and msg.usage_metadata:
            total_tokens += msg.usage_metadata.get("total_tokens", 0)

    print(f"\n🔥 Total Tokens Spent: {total_tokens}")
    
    # Extract the final output from the message history
    final_message = response["messages"][-1].content
    
    print("\n🔥 FINAL RIRI ANSWER 🔥\n")
    print(textwrap.fill(final_message, width=80))

if __name__ == "__main__":
    asyncio.run(main())