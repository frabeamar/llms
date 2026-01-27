from dataclasses import dataclass
from langchain_core.exceptions import OutputParserException
import tqdm
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from my_log import setup_logging
import logging
log = setup_logging(__name__)
"""
Groundedness measures how much of the model's response is derived strictly from the retrieved context.
Relevance: Cosine Similarity between the embedding of the query ($q$) and the embedding of the response ($r$):
standalone:nformation Gain or the semantic overlap between the re-written query ($q_{new}$) and the original conversation thread ($C$).A common analytical approach is to measure the Jaccard Similarity Index between the sets of key entities/terms in the standalone query ($S$) and the conversation context ($C$): 
basically if the question can be answere
"""
question_groundedness_critique_prompt = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
"""

question_relevance_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to machine learning developers building NLP applications with the Hugging Face ecosystem.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.


Now here is the question.

Question: {question}\n
"""

question_standalone_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independent this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.

Now here is the question.

Question: {question}\n
"""


class TotalRating(BaseModel):
    total_rating: int = Field(
        description="Total Rating score (your rating, as only a number between 1 and 5), without additional text"
    )
    evaluation: str = Field(
        description="Evaluation::  (your rationale for the rating, as a text)"
    )


@dataclass
class EvalBot:
    chain: RunnableSerializable

    @classmethod
    def with_prompt(cls, prompt):
        parser = StrOutputParser()
        chat = ChatOllama(model="llama3.2")

        parser = PydanticOutputParser(pydantic_object=TotalRating)

        prompt = ChatPromptTemplate.from_template(
            template=prompt,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | chat.with_structured_output(TotalRating)
            # | parser
        )

        return EvalBot(chain)


def evaluate(outputs: list[dict]):
    bots = {
        "groundedness": EvalBot.with_prompt(question_groundedness_critique_prompt),
        "relevance": EvalBot.with_prompt(question_relevance_critique_prompt),
        "standalone": EvalBot.with_prompt(question_standalone_critique_prompt),
    }
    log.info("Generating critique for each QA couple...")
    for output in tqdm.tqdm(outputs):
        for criterion, bot in bots.items():
            try:
                answer = bot.chain.invoke(output)
                
                output.update(
                    {
                        f"{criterion}_score": answer.total_rating,
                        f"{criterion}_eval": answer.evaluation,
                    }
                )
            except OutputParserException as e:
                print(e)
                continue

    return outputs
