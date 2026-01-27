import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
import nest_asyncio
import pandas as pd
import seaborn as sns
import tqdm
from langchain.indexes import SQLRecordManager, index
from langchain_chroma import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
from ragas.cache import DiskCacheBackend
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)
from typer import Typer

from evaluate_llms import evaluate
from my_log import setup_logging

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
nest_asyncio.apply()

setup_logging(__name__)
dotenv.load_dotenv(".env")
app = Typer(pretty_exceptions_show_locals=False)
log = logging.getLogger(__name__)

DATA = Path.home() / "data"
DIVIDER = "+++++"


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
            persist_directory=str(DATA / "chroma_db"),
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        record_manager = SQLRecordManager(
            cls.namespace, db_url=f"sqlite:///{DATA}/record_manager_cache.sql"
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
        return [[d.page_content for d in f] for f in found]


@dataclass
class ChatBot:
    chain: RunnableSerializable

    @classmethod
    def default(cls):
        template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """
        prompt = ChatPromptTemplate.from_template(
            template=template,
        )

        parser = StrOutputParser()
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | ChatOllama(model="llama3.2")
            | parser
        )

        return ChatBot(chain)


def rags() -> None:
    # load and prepare datasets
    docs = []
    for doc in [
        "multi_passage_answer_questions",
        "single_passage_answer_questions",
        "no_answer_questions",
    ]:
        df = pd.read_csv(DATA / "small_rag_dataset" / f"{doc}.csv")
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

    documents = pd.read_csv(DATA / "small_rag_dataset/documents.csv")
    docs = DataFrameLoader((documents)).load()
    pipeline = TextPipeline.default()
    pipeline.load_incrementally(docs)

    chatbot = ChatBot.default()

    context = pipeline.query(list(df.question))
    data = [
        {"context": c, "question": p, "expected_answer": a}
        for c, p, a in zip(context, df.question, df.expected_answer)
    ]
    answer = chatbot.chain.batch(data)
    evals = evaluate(data)
    df_evals = pd.DataFrame(evals)
    df_evals["answer"] = answer
    df_evals["context"] = [DIVIDER.join(d["context"]) for d in data]
    df_evals.to_csv("evals.csv", index=False)
    ## need to check if I get numbers


async def ragas_metrics():
    ollama_client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Ollama doesn't check keys, but Ragas/OpenAI client requires a string
    )

    evaluator_llm = llm_factory(model="llama3", client=ollama_client)

    log.info("Faithfulness score")
    # results = evaluate(dataset=my_dataset, metrics=[...], llm=evaluator_llm))

    df_evals = pd.read_csv("evals.csv")

    cache = DiskCacheBackend(cache_dir=DATA / "my_eval_cache")
    embeddings = embedding_factory(
        provider="openai",
        model="mxbai-embed-large",  # Replace with the model you 'ollama pull'ed
        client=ollama_client,
        cache=cache,
    )
    # --- RETRIEVER METRICS ---
    # Evaluates the quality of the search results (Context)

    # Measures if the most relevant chunks are ranked at the top of the results.
    precision = ContextPrecision(llm=evaluator_llm)
    # Measures if the retrieved context contains enough info to reach the ground truth.
    recall = ContextRecall(llm=evaluator_llm)
    # A specialized recall that checks if key entities (names, dates, places)
    # from the reference are present in the retrieved context.
    entity_recall = ContextEntityRecall(llm=evaluator_llm)

    # --- GENERATOR METRICS ---
    # Evaluates the quality of the AI's generated response (Answer)

    # Measures "Groundedness": Does the answer stay true to the retrieved context
    # and avoid hallucinations?
    faithfulness = Faithfulness(llm=evaluator_llm)
    # Measures "Truth": How well does the answer match the human-verified
    # ground truth (Reference)?
    factual_correctness = FactualCorrectness(llm=evaluator_llm)
    # Measures "Helpfulness": Does the answer directly address the user's
    # question, regardless of the source?
    answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=embeddings)
    df_evals = pd.read_csv("evals.csv")
    scores = []

    log.info("Start evaluations")
    for i, row in tqdm.tqdm(
        df_evals.iterrows(), "Evaluate the RAG system", total=len(df_evals)
    ):
        user_input = row.question
        response = row.answer
        reference = row.expected_answer
        context = row.context.split(DIVIDER)

        score = {}

        score["answer_relevancy"] = await answer_relevancy.ascore(
            user_input=user_input,
            response=response,
        )

        score["faithfullness"] = await faithfulness.ascore(
            user_input=user_input,
            response=response,
            retrieved_contexts=context,
        )  # should be a list of strings

        score["context_precision"] = await precision.ascore(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=context,
        )
        score["context_recall"] = await recall.ascore(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=context,
        )
        score["entity_recall"] = await entity_recall.ascore(
            reference=response, retrieved_contexts=context
        )

        score["factual_correctness"] = await factual_correctness.ascore(
            reference=reference,
            response=response,
        )

        score = {s: c.value for s, c in score.items()}
        scores.append(score)
        # seems to be rather low. Might need a better judge
    pd.DataFrame(scores).to_csv("scores.csv")


def plot():
    melted = pd.melt(
        pd.read_csv("evals.csv"),
        id_vars=[
            "context",
            "question",
            "relevance_score",
            "relevance_eval",
            "standalone_score",
            "standalone_eval",
        ],
    )
    sns.displot(melted, x="value", kind="hist", row="variable")
    plt.savefig("scores")


@app.command()
def main():
    # rags()
    asyncio.run(ragas_metrics())


app()
