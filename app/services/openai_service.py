import os
import json
import requests
from openai import OpenAI

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_EMBEDDING_MODEL = 'text-embedding-ada-002'

def get_embedding(chunk):
  url = 'https://api.openai.com/v1/embeddings'
  headers = {
      'content-type': 'application/json; charset=utf-8',
      'Authorization': f"Bearer {OPENAI_API_KEY}"
  }
  data = {
      'model': OPENAI_EMBEDDING_MODEL,
      'input': chunk
  }
  response = requests.post(url, headers=headers, data=json.dumps(data))
  response_json = response.json()
  embedding = response_json["data"][0]["embedding"]
  return embedding

# Now you can access your API key or other variables
api_key = os.getenv('OPENAI_API_KEY')


# add this to the top of the file under the imports
CHATGPT_MODEL = 'gpt-3.5-turbo-0125'

def get_llm_answer(prompt):
    # Aggregate a messages array to send to the LLM
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": prompt})
    # Send the payload to the LLM to retrieve an answer
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'content-type': 'application/json; charset=utf-8',
        'Authorization': f"Bearer {api_key}"
    }
    data = {
        'model': CHATGPT_MODEL,
        'messages': messages,
        'temperature': 1,
        'max_tokens': 4096
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # return the final answer
    response_json = response.json()
    completion = response_json["choices"][0]["message"]["content"]
    return completion