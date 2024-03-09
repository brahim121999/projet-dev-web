from . import api_blueprint
from flask import request, jsonify
from app.services import openai_service, pinecone_service, scraping_service
from app.utils.helper_functions import chunk_text, build_prompt

# Sample index name since we're only creating a single index
PINECONE_INDEX_NAME = 'bob3'

@api_blueprint.route('/embed-and-store', methods=['POST'])
def embed_and_store():
    # handles scraping the URL, embedding the texts, and
    # uploading to the vector database.
    url = request.json['url']
    url_text = scraping_service.scrape_website(url)
    chunks = chunk_text(url_text)
    pinecone_service.embed_chunks_and_upload_to_pinecone(chunks, PINECONE_INDEX_NAME)
    response_json = {
        "message": "Chunks embedded and stored successfully"
    }
    return jsonify(response_json)


@api_blueprint.route('/handle-query', methods=['POST'])
def handle_query():
  # handles embedding the user's question,
  # finding relevant context from the vector database,
  # building the prompt for the LLM,
  # and sending the prompt to the LLM's API to get an answer.
  question = request.json['question']
  context_chunks = pinecone_service.get_most_similar_chunks_for_query(question, PINECONE_INDEX_NAME)
  #print(context_chunks)
  prompt = build_prompt(question, context_chunks)
  #print(prompt)
  answer = openai_service.get_llm_answer(prompt)

  print(answer)

  #with open("test.txt", "w") as my_file:
  #  my_file.write(answer)


  return jsonify({"question": question, "answer": answer})
