from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_chroma import Chroma
from time import sleep
import tqdm
import hashlib
load_dotenv(".env")

def load_gemini_chat():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def load_gemini_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")



def load_ollama_chat():
    return ChatOllama(model = "llama3.1")

def hash_id(text):
    return hashlib.md5(text.encode()).hexdigest()

def save_embedding_locally(all_splits:list[Document], vector_store:Chroma):
    """
    Save embeddings to local vector store, avoiding duplicates
    Avoid rate limits from the google API by batching and sleeping
    """
    hashed_splits = {hash_id(s.page_content): s for s in all_splits}
    # avoid duplicate docs
    hashed_splits = list(hashed_splits.items())
    for i in tqdm.tqdm(range(1, len(hashed_splits), 10), total = len(hashed_splits) // 10):
        batch = hashed_splits[i : i + 10]
        ids, docs = zip(*batch)
        added_ids = set(vector_store.get()["ids"])
        if  len(set(ids) - set(added_ids)) == 0:
            continue
        else:
            batch = [(id, doc) for id, doc in batch if id not in added_ids]
            ids, docs = zip(*batch)

        sleep(10)  # to avoid rate limits
        vector_store.add_documents(documents=list(docs), ids=ids)

