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
                    "description": "the message that needs to be send",
                },
            },
            "required": ["name", "message"],
        },
    },
]


system_prompt = 'you are a cute helpful asistant called Riri and you respond in a very funny way'
user_prompt = 'What is the price of bitcoin and send it to peter'
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
    print(f"Riri decided to call tool: {item.name} with args: {item.arguments}")
# -----------------------------------------------


stop = False
while True:
    print("\n--- MODEL OUTPUT ---")
    print(response.output)
    print(json.dumps(response.model_dump(), indent=2))
    print('Token Usage: ',response.usage.total_tokens)
    base_message += response.output

    for item in response.output:

        if item.type == "function_call":
            name = item.name
            args = json.loads(item.arguments)
            call_id = item.call_id

            if name == "get_price":
                result = get_price(**args)
            elif name == "send_email":
                result = send_email(**args)
            else:
                result = f"Unknown tool: {name}"

            base_message.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result)
            })

            response = client.responses.create(
                model="gpt-4o-mini",
                input=base_message,
                tools=tools
            )
            #print(json.dumps(response.model_dump(), indent=2))

            break  

        elif item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    print("\n🔥 FINAL RIRI ANSWER 🔥\n")
                    print(c.text)
                    stop = True
                    break
            break
        else:
            print("⚠️ Unexpected item:", item)
    if stop:
        break

        

print ('finally')




