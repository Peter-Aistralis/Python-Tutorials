import json
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import requests


OPENAI_KEY = os.getenv("Openai_Key")
CMC_KEY = os.getenv("CMC_Key")

client = OpenAI(api_key=OPENAI_KEY)

def print_reasoning(response):
    print("\n🧠 --- MODEL REASONING ---")
    has_reasoning = False
    for item in response.output:
        if item.type == "reasoning":
            for s in item.summary:
                print(s.text)
                has_reasoning = True
    if not has_reasoning:
        print("(No readable reasoning summary returned this turn)")
    print("--------------------------\n")

def call_function(name, args):
    if name == "get_price":
        return get_price(**args)
    if name == "send_email":
        return send_email(**args)
    if name == "get_temperature":
        return get_temperature(**args)
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
    # Just an empty mock function
    return f'Email has been sent to {name}'

def get_temperature(city):
    # Another mock function
    return f'temperature in {city} is 21 degrees'


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
    {
        "type": "function",
        "name": "get_temperature",
        "description": "get the temperature of a specific city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "name of the city",
                },
            },
            "required": ["city"],
        },
    },
]


system_prompt = 'you are a cute helpful asistant called Kiky and you respond in a very funny way'
user_prompt = 'What is the price of bitcoin and Cardano. Send the price of Bitcoin to Peter and the price of Cardano to Roger together with the temperature in Ghent'
#model_name="gpt-4o-mini"
model_name = "gpt-6-luna"
base_message = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
    ]
 

response = client.responses.create(
    model=model_name,
    tools = tools,
    input=base_message,
    reasoning={"effort": "medium", "summary": "auto"} # <--- Added to trigger summary
)
print_reasoning(response)

while True:
    base_message += response.output

    function_calls = [i for i in response.output if i.type == "function_call"]

    # No more tool calls → model is done, print the message
    if not function_calls:
        print ('Final response from Kiky')
        print ('------------------------')
        print(response.output_text)
        print('\nToken Usage: ',response.usage.total_tokens)
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
    

    
        





