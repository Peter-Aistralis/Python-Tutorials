import json
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import requests
from pprint import pprint

OPENAI_KEY = os.getenv("Openai_Key")
CMC_KEY = os.getenv("CMC_Key")

client = OpenAI(api_key=OPENAI_KEY)

def call_function(name, args):
    if name == "get_price":
        return get_price(**args)
    if name == "send_email":
        return send_email(**args)
    raise ValueError(f"Unknown function: {name}")

def get_price(symbol):
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


def send_email(name, message):
    #something
    #print ('in email function')
    return f'Email has been sent to {name}'


tools = [
    {
        "type": "function",
        "name": "get_price",
        "description": "Get the current price for a cryptocurrency.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The symbol of the cryptocurrency",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "type": "function",
        "name": "send_email",
        "description": "send a notification or message to a person",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "name of the person to notify",
                },
                "message": {
                    "type": "string",
                    "description": "nthe message that needs to be send",
                },
            },
            "required": ["name", "message"],
        },
    },
]


system_prompt = 'you are a cute helpful asistant called Riri'
user_prompt = 'What is the price of bitcoin and Cardano. Send the price of Bitcoin to Peter the the price of Cardano to XXX'
model_name="gpt-4o-mini"
#model_name = "o3-mini"
base_message = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
    ]
 

response = client.responses.create(
    model=model_name,
    tools = tools,
    input=base_message
)

print("\n--- INITIAL MODEL OUTPUT (FIRST TURN) ---")
print(json.dumps(response.model_dump(), indent=2))
print(response.output)
for item in response.output:
  if item.type == "message":
    for c in item.content:
      if c.type == "output_text":
        print("Riri initially said:", c.text)
  elif item.type == "function_call":
    print(f"decided to call tool: {item.name} with args: {item.arguments}")
# -----------------------------------------------



while True:
    base_message += response.output

    function_calls = [i for i in response.output if i.type == "function_call"]

    # No more tool calls → model is done, print the message
    if not function_calls:
        print("\nFinal Answer \n")
        print(response.output_text)
        break

    # Execute ALL calls in this turn (handles parallel + sequential)
    for tool_call in function_calls:
        name = tool_call.name
        args = json.loads(tool_call.arguments)
        result = call_function(name, args)
        base_message.append({
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": json.dumps(result),
        })

    # Re-ask ONCE after collecting all outputs for this turn
    response = client.responses.create(model=model_name, input=base_message, tools=tools)
    print(json.dumps(response.model_dump(), indent=2))
    
        





