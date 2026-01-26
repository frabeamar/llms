from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import HypotheticalDocumentEmbedder
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from time import sleep
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
import dotenv
from numpy import single
from ollama import chat
import pandas as pd
import torch
from langchain.indexes import SQLRecordManager, index
from langchain_chroma import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents import Document
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tqdm
from evaluate_llms import evaluate
dotenv.load_dotenv(".env")


DATA = Path("small_rag_dataset")


@dataclass
class TextPipeline:
    vector_store: Chroma
    retriever: VectorStoreRetriever
    record_manager: SQLRecordManager
    embeddings: GoogleGenerativeAIEmbeddings
    chat: ChatGoogleGenerativeAI
    text_splitter: RecursiveCharacterTextSplitter
    namespace: str = "my_docs"

    @classmethod
    def default(cls):
        # embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")

        chat = ChatOllama(model="llama3.2")
        vector_store = Chroma(
            collection_name="documents",
            embedding_function=embeddings,
            persist_directory="./chroma_db",
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        record_manager = SQLRecordManager(
            cls.namespace, db_url="sqlite:///record_manager_cache.sql"
        )

        record_manager.create_schema()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        return cls(
            vector_store, retriever, record_manager, embeddings, chat, text_splitter
        )

    def load_incrementally(self, docs: list[Document], N: int = 10):
        all_splits = self.text_splitter.split_documents(documents=docs)
        for i in tqdm.tqdm(range(0, len(all_splits) // N)):
            result = index(
                all_splits[i * N : (i + 1) * N],
                self.record_manager,
                self.vector_store,
                cleanup="incremental",
                source_id_key="start_index",
            )

    def query(self, query: list[str]):
        found = self.retriever.batch(query)
        return [[d.page_content for d in f ] for f in found ]

    @cached_property
    def hypothetical_document_embedder(self):
        # Create the HyDE embedder
        return  HypotheticalDocumentEmbedder.from_llm(
            self.chat, 
            self.embeddings, 
            "web_search" # Pre-defined prompt style
        )

    def hypo_query_search(self, query:list[str]):
        ret = self.vector_store.as_retriever(search_type= "mmr", k=20)
        found = [f.page_content for f in ret.invoke(query[0])]
        self.vector_store.similarity_search(query[0])
        return self.query(augmented_query)

@dataclass
class ChatBot:
    chain: RunnableSerializable

    @classmethod
    def default(cls, text_pipeline:TextPipeline):
        template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """
        return cls.with_prompt(template,text_pipeline)
    
    @classmethod
    def with_prompt(cls, prompt,text_pipeline:TextPipeline):
        parser = StrOutputParser()

        prompt = ChatPromptTemplate.from_template(prompt)
        
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | text_pipeline.chat
            | parser
        )
        return ChatBot(chain)



def main() -> None:
    torch.mps.set_per_process_memory_fraction(0.0)
    # set_per_process_memory_fraction: max gpu memory a model can take

    # load and prepare datasets
    docs = []
    for doc in [
        "multi_passage_answer_questions",
        "single_passage_answer_questions",
        "no_answer_questions",
    ]:
        df = pd.read_csv(DATA / f"{doc}.csv")
        if doc == "no_answer_questions":
            df["expected_answer"] = (
                "The answer to your question is not in the provided text."
            )
        else:
            df = df.rename(columns={"answer": "expected_answer"})

        df = df.assign(doc_type=doc)
        df["actual_answer"] = None
        docs.append(df)

    df = pd.concat(docs)

    documents = pd.read_csv(DATA / "documents.csv")
    docs = DataFrameLoader((documents)).load()
    pipeline = TextPipeline.default()

    chatbot = ChatBot.default(pipeline)
    single_passage = df[df.doc_type == "single_passage_answer_questions"].question

    
    context = pipeline.query(list(single_passage))
    data = [{"context":c, "question":p} for c, p in zip(context, single_passage)]
    answer = chatbot.chain.batch(data)
    evals = evaluate(data, pipeline.chat)
    breakpoint()



main()
