import os
import chainlit as cl
from dotenv import load_dotenv
from langchain.vectorstores import Vectara
from sentence_transformers import CrossEncoder
load_dotenv()
import requests
import json

vectara_customer_id = os.getenv("VECTARA_CUSTOMER_ID")
vectara_corpus_id = 2
vectara_api_key = 'zqt_Y3kD9bueJq3QO5t_FISVQLmgTWMDhzgMgK9Isw'

# Input your API keys in .env
vectara_instance = Vectara(
    vectara_customer_id=os.getenv("VECTARA_CUSTOMER_ID"),
    vectara_corpus_id=2,
    vectara_api_key='zqt_Y3kD9bueJq3QO5t_FISVQLmgTWMDhzgMgK9Isw',
)

config = {
    "api_key": str(vectara_api_key),
    "customer_id": str(vectara_customer_id),
    "corpus_id": str(vectara_corpus_id),
    "lambda_val": 0.025,
    "top_k": 10,
}

model = CrossEncoder('vectara/hallucination_evaluation_model')

@cl.on_message
async def main(message):
    corpus_key = [
        {
            "customerId": config["customer_id"],
            "corpusId": config["corpus_id"],
            "lexicalInterpolationConfig": {"lambda": config["lambda_val"]},
        }
    ]
    data = {
        "query": [
            {
                "query": message,
                "start": 0,
                "numResults": config["top_k"],
                "contextConfig": {
                    "sentencesBefore": 2,
                    "sentencesAfter": 2,
                },
                "corpusKey": corpus_key,
                "summary": [
                    {
                        "responseLang": "eng",
                        "maxSummarizedResults": 5,
                    }
                ]
            }
        ]
    }

    headers = {
        "x-api-key": config["api_key"],
        "customer-id": config["customer_id"],
        "Content-Type": "application/json",
    }
    response = requests.post(
        headers=headers,
        url="https://api.vectara.io/v1/query",
        data=json.dumps(data),
    )
    if response.status_code != 200:
        print(
            "Query failed %s",
            f"(code {response.status_code}, reason {response.reason}, details "
            f"{response.text})",
        )
        return []

    result = response.json()
    responses = result["responseSet"][0]["response"]
    documents = result["responseSet"][0]["document"]
    summary = result["responseSet"][0]["summary"][0]["text"]

    res = [[r['text'], r['score']] for r in responses]
    texts = [r[0] for r in res[:5]]
    scores = [model.predict([text, summary]) for text in texts]

    text_elements = []
    source_names = []
    docs = vectara_instance.similarity_search(message)
    for source_idx, source_doc in enumerate(docs[:5]):
        source_name = f"Source {source_idx + 1}"
        text_elements.append(
            cl.Text(content=source_doc.page_content, name=source_name)
        )
    source_names = [text_el.name for text_el in text_elements]


    ans = f"{summary}\n Sources: {', '.join(source_names)} \n HHEM Scores: {scores}"
    
    await cl.Message(content=ans, author="Assistant", elements=text_elements).send()
