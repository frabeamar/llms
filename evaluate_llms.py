import asyncio
import os

import dotenv
from google import genai
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import AnswerRelevancy, DiscreteMetric, Faithfulness

from model import load_gemini_chat

dotenv.load_dotenv(".env")
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


client = load_gemini_chat()
llm = llm_factory("gemini-2.0-flash", client=client)
# Adapter is auto-detected as "litellm" for google provider
# 3. Run Evaluation
# Ragas will automatically use Google Embeddings for metrics that require them
metrics = [Faithfulness(llm=llm), AnswerRelevancy(llm=llm)]
results = evaluate(my_dataset, metrics=metrics)
# Create a custom aspect evaluator
metric = DiscreteMetric(
    name="summary_accuracy",
    allowed_values=["accurate", "inaccurate"],
    prompt="""Evaluate if the summary is accurate and captures key information.

Response: {response}

Answer with only 'accurate' or 'inaccurate'.""",
)


# Score your application's output
async def main():
    score = await metric.ascore(llm=llm, response="The summary of the text is...")
    print(f"Score: {score.value}")  # 'accurate' or 'inaccurate'
    print(f"Reason: {score.reason}")


if __name__ == "__main__":
    asyncio.run(main())
