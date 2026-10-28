import json
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import requests


OPENAI_KEY = os.getenv("Openai_Key")
CMC_KEY = os.getenv("CMC_Key")

client = OpenAI(api_key=OPENAI_KEY)

def get_price(symbol):
    url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest'
    params = {
        "symbol": symbol
    }
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": CMC_KEY
    }
    response_cmc = requests.get(url, headers=headers, params=params)
    data = response_cmc.json()
    print ('in function, the price of bitcoin is:', data["data"][symbol][0]["quote"]["USD"]["price"])
    return data["data"][symbol][0]["quote"]["USD"]["price"]


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
]



system_prompt = 'you are a cute helpful asistant called Riri and you answer is professional way'
user_prompt = 'What is the price of bitcoin and cardano?'
model_name="gpt-4o-mini"

base_message = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
    ]
 

response_llm = client.responses.create(
    model=model_name,
    tools = tools,
    input=base_message
)
print ('ouput llm call 1')
#print(response_llm.output_text)
print(json.dumps(response_llm.model_dump(), indent=2))


if response_llm.output[0].type == "function_call":
    tool_call_object = response_llm.output[0]
    args = json.loads(tool_call_object.arguments)
    print ('args = ', args)
    symbol = args.get("symbol") 
    
    price = get_price(symbol)
    
    response2_llm = client.responses.create(
        model = model_name,
        input = [
            *base_message,
            {
                "role": "user",
                "content": f"The tool `get_price` returned that the price of {symbol} is ${price:.2f}.",
            },
        ],
    )
    
    
print ('ouput llm call 2')
print(json.dumps(response2_llm.model_dump(), indent=2))    
print("\n--- Final Riri Response ---")
#print(json.dumps(follow_up.model_dump(), indent=2))
print(response2_llm.output[0].content[0].text)
print('Token Usage: ',response2_llm.usage.total_tokens)

