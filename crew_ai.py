# pip install crewai langchain-openai

import logging
import os

from crewai import Agent, Crew, Task
from crewai.tools import tool
from dotenv import load_dotenv

from google_adk import GEMINI_MODEL
# --- Load  API Key ---
load_dotenv(".env")

# --- Best Practice: Configure Logging ---
# A basic logging setup helps in debugging and tracking the crew's execution.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)




# --- 1. Refactored Tool: Returns Clean Data ---
# The tool now returns raw data (a float) or raises a standard Python error.
# This makes it more reusable and forces the agent to handle outcomes properly.
@tool("Stock Price Lookup Tool")
def get_stock_price(ticker: str) -> float:
    """
    Fetches the latest simulated stock price for a given stock ticker symbol.
    Returns the price as a float. Raises a ValueError if the ticker is not found.
    """
    logging.info(f"Tool Call: get_stock_price for ticker '{ticker}'")
    simulated_prices = {
        "AAPL": 178.15,
        "GOOGL": 1750.30,
        "MSFT": 425.50,
    }
    price = simulated_prices.get(ticker.upper())

    if price is not None:
        return price
    else:
        # Raising a specific error is better than returning a string.
        # The agent is equipped to handle exceptions and can decide on the next action.
        raise ValueError(f"Simulated price for ticker '{ticker.upper()}' not found.")

def tool_calling():

    # --- 2. Define the Agent ---
    financial_analyst_agent = Agent(
        role="Senior Financial Analyst",
        goal="Analyze stock data using provided tools and report key prices.",
        backstory="You are an experienced financial analyst adept at using data sources to find stock information. You provide clear, direct answers.",
        verbose=True,
        llm = GEMINI_MODEL, # defaults to open model if not specified
        tools=[get_stock_price],
        allow_delegation=False, # prohibit calling other agents; tools are allowed
        # not sure what the other agents would be since none are defined
    )

    # --- 3. Refined Task: Clearer Instructions and Error Handling ---
    # The task description is more specific and guides the agent on how to react
    # to both successful data retrieval and potential errors.
    analyze_aapl_task = Task(
        description=(
            "What is the current simulated stock price for Apple (ticker: AAPL)? "
            "Use the 'Stock Price Lookup Tool' to find it. "
            "If the ticker is not found, you must report that you were unable to retrieve the price."
        ),
        expected_output=(
            "A single, clear sentence stating the simulated stock price for AAPL. "
            "For example: 'The simulated stock price for AAPL is $178.15.' "
            "If the price cannot be found, state that clearly."
        ),
        agent=financial_analyst_agent,
    )

    # --- 4. Formulate the Crew ---
    # The crew orchestrates how the agent and task work together.
    financial_crew = Crew(
        agents=[financial_analyst_agent],
        tasks=[analyze_aapl_task],
        verbose=True,  # Set to False for less detailed logs in production
    )
    """Main function to run the crew."""

    print("\n## Starting the Financial Crew...")
    print("---------------------------------")

    # The kickoff method starts the execution.
    result = financial_crew.kickoff()

    print("\n---------------------------------")
    print("## Crew execution finished.")
    print("\nFinal Result:\n", result)


# --- 5. Run the Crew within a Main Execution Block ---


if __name__ == "__main__":
    tool_calling()
