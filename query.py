# query.py
import os
from openai import OpenAI
import chromadb
chroma_client = chromadb.PersistentClient(path="./chroma_db")

from utils import read_file

API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=API_KEY)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection('mzu_knowledge')

SYSTEM_PROMPT = (
    "You are an assistant for Mizoram University (MZU). Answer the user's question using ONLY the provided context. "
    "If the answer is not in the context, say you don't know and suggest a way to find it. Keep answers concise and factual."
)


def answer_query(query, k=3):
    qresp = client.embeddings.create(model='text-embedding-3-small', input=query)
    qemb = qresp.data[0].embedding

    results = collection.query(query_embeddings=[qemb], n_results=k)
    docs = results['documents'][0]
    metadatas = results.get('metadatas', [[]])[0]

    context = '\n\n'.join([f"[Source: {m.get('source','mzu')} chunk:{m.get('chunk')}]\n" + d for d, m in zip(docs, metadatas)])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    resp = client.chat.completions.create(model='gpt-4o', messages=messages, max_tokens=400)
    return resp.choices[0].message.content

if __name__ == '__main__':
    q = input('Question: ')
    print(answer_query(q))
