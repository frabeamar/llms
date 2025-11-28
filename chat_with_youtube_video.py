import os
import tempfile

import dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import typer
from google import genai 
from pytubefix import YouTube
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_chroma import Chroma
from model import load_gemini_chat, load_gemini_embeddings, save_embedding_locally
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch


dotenv.load_dotenv(".env")

app = typer.Typer(pretty_exceptions_show_locals=False)
client = genai.Client()
YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=cdiD-9MMpb0"  # your video


@app.command()
def download_and_transcribe(url: str = YOUTUBE_VIDEO):
    if not os.path.exists("transcription.txt"):
        youtube = YouTube(YOUTUBE_VIDEO)
        audio = youtube.streams.filter(only_audio=True).first()

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_file = audio.download(output_path=tmpdir)

            # Upload audio to Gemini
            uploaded = client.files.upload(file=audio_file)

            # Ask Gemini to transcribe
            prompt = "Generate a transcript of the speech in this audio file."
            response = client.models.generate_content(
                model="gemini-2.5-flash",  # or a supported Gemini audio model
                contents=[prompt, uploaded],
            )
            transcription = response.text.strip()

            with open("transcription.txt", "w", encoding="utf-8") as f:
                f.write(transcription)

    print("Done — transcription.txt written.")

@app.command()
def chat_with_video():

    loader = TextLoader("transcription.txt")
    text_documents = loader.load()
    parser = StrOutputParser()
    model = load_gemini_chat()
 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents = text_splitter.split_documents(text_documents)
    embeddings = load_gemini_embeddings()
    vector_store = Chroma(
        collection_name="youtube_video_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_youtube_db",
    )
    save_embedding_locally(vector_store=vector_store, all_splits=documents)
    #RunnablePassthrough copies the input over
    template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | model
        # | parser
    )

    no_parser = (
        {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | model
    )
    chain = no_parser | parser
    # only gemini pro can answer the question
    res = chain.invoke("What is synthetic intelligence?")
    print(res)

app()
