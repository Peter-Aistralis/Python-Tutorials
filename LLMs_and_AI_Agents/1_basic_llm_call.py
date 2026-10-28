import json
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()



OPENAI_KEY = os.getenv("Openai_Key")

client = OpenAI(api_key=OPENAI_KEY)

system_prompt = 'you are a cute helpful asistant called Riri and you respond in a very funny way'
user_prompt = 'what is the abbreviation of bitcoin and what is the latest price you are aware of'
model_name="gpt-4o-mini"

base_message = [
    {"role": "system", 
     "content": system_prompt},
    {"role": "user", 
     "content": user_prompt}
    ]
 

response = client.responses.create(
    model=model_name,
    input=base_message
)

print(json.dumps(response.model_dump(), indent=2))
print ('-----')
print(response.output[0].content[0].text)
print('Token Usage: ',response.usage.total_tokens)
print(response.output_text)
