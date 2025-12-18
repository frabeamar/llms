from pathlib import Path
from time import sleep
from typing import Any
import urllib
import bs4
import dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    dynamic_prompt,
)
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_ollama import ChatOllama, OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typer import Typer
from llms.model import load_gemini_embeddings

from model import save_embedding_locally
dotenv.load_dotenv(".env")
app = Typer(pretty_exceptions_show_locals=False)

@app.command()
def retrieval():
    filepath = Path("nke.pdf")
    if not filepath.exists():
        url = "https://raw.githubusercontent.com/langchain-ai/langchain/v0.3/docs/docs/example_data/nke-10k-2023.pdf"
        urllib.request.urlretrieve(url, "nke.pdf")

    # load pdf page by page
    loader = PyPDFLoader(filepath)
    docs = loader.load()

    # split by newline and character count, have some overlap to avoid losing context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)


    # this has rate limits
    embeddings = load_gemini_embeddings()

    # create vector store, a database, can be local or remote
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )

    save_embedding_locally(vector_store=vector_store, all_splits=all_splits)

    # there are different ways to write a retriever
    # vector store does not implement invoke and batch
    @chain
    def retriever(query: str) -> list[Document]:
        return vector_store.similarity_search(query, k=1)

    # vector store can give back a retriever
    ret = vector_store.as_retriever(
        search_type="similarity",  # mmr or similarity_score_threshold
        search_kwargs={"k": 1},
    )
    # single result
    results = vector_store.similarity_search(
        "How many distribution centers does Nike have in the US?"
    )
    results = ret.batch( # or invoke, synchronous, batch async
        [
            "How many distribution centers does Nike have in the US?",
            "When was Nike incorporated?",
        ],
    )
    for doc in results[0]:
        print(doc.page_content)
    
def load_blog():
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    return all_splits


@app.command()
def rag():


    # define a model and a vetor store

    model = ChatOllama(model="llama3.1")
    embeddings = OllamaEmbeddings(model="llama3.1")
    # create vector store, a database, can be local or remote
    # vector_store = InMemoryVectorStore(embeddings) #this cannot be pickled
    vector_store = Chroma(
        collection_name="rags",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )

    all_splits = load_blog()

    save_embedding_locally(vector_store=vector_store, all_splits=all_splits)

    # Construct a tool for retrieving context
    # here the llms decides when to use the tool
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=10)
  
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    tools = [retrieve_context]
    # If desired, specify custom instructions
    prompt = (
        "You have access to a tool that retrieves context from a blog post. "
        "Use the tool to help answer user queries."
    )
    agent = create_agent(model, tools, system_prompt=prompt)
    # query = "What is task decomposition?"
    query = (
        "What is the standard method for Task Decomposition? with llms\n\n"
        "Once you get the answer, look up common extensions of that method."
    )
    # does not really work, the query gets augmented with 'project management'
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

    # in this case we search first and then ask the llms for answer
    # giving the retrived context together with the query
    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        """Inject context into state messages."""
        last_query = request.state["messages"][-1].text
        retrieved_docs = vector_store.similarity_search(last_query)

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        system_message = (
            "You are a helpful assistant. Use the following context in your response:"
            f"\n\n{docs_content}"
        )

        return system_message

    agent = create_agent(model, tools=[], middleware=[prompt_with_context])

    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


# retrieve + concat to query can also be done with a class


class State(AgentState):
    context: list[Document]


class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
    state_schema = State

    def __init__(self, vector_store: Chroma) -> None:
        self.vector_store = vector_store

    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        ret  = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k":10})
        retrieved_docs = ret.invoke(last_message.text)

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        augmented_message_content = (
            f"{last_message.text}\n\n"
            "Use the following context to answer the query:\n"
            f"{docs_content}"
        )
        return {
            "messages": [
                last_message.model_copy(update={"content": augmented_message_content})
            ],
            "context": retrieved_docs,
        }



@app.command()
def rag_middleware():
    vector_store = Chroma(
        collection_name="rags",
        embedding_function=OllamaEmbeddings(model="llama3.1"),
        persist_directory="./chroma_langchain_db",
    )
    save_embedding_locally(vector_store=vector_store, all_splits=load_blog())
    all_docs = vector_store.get(include=["documents"])
    print("Documents in vector store",  len(all_docs["documents"]))
    chatbot = OllamaLLM(model="llama3.1")
    agent = create_agent(
        chatbot,
        tools=[],
        middleware=[RetrieveDocumentsMiddleware(vector_store=vector_store)],
    )   
    for step in agent.stream(
        {"messages": [{"role": "user", "content": "Explain task decomposition."}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


app()
