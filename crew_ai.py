# pip install crewai langchain-openai

import logging

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


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
        llm=GEMINI_MODEL,  # defaults to open model if not specified
        tools=[get_stock_price],
        allow_delegation=False,  # prohibit calling other agents; tools are allowed
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


def planning():
    # 2. Define a clear and focused agent
    planner_writer_agent = Agent(
        role="Article Planner and Writer",
        goal="Plan and then write a concise, engaging summary on a specified topic.",
        backstory=(
            "You are an expert technical writer and content strategist. "
            "Your strength lies in creating a clear, actionable plan before writing, "
            "ensuring the final summary is both informative and easy to digest."
        ),
        verbose=True,
        allow_delegation=False,
        llm=GEMINI_MODEL,  # Assign the specific LLM to the agent
    )

    # 3. Define a task with a more structured and specific expected output
    topic = "The importance of Reinforcement Learning in AI"
    high_level_task = Task(
        description=(
            f"1. Create a bullet-point plan for a summary on the topic: '{topic}'.\n"
            f"2. Write the summary based on your plan, keeping it around 200 words."
        ),
        expected_output=(
            "A final report containing two distinct sections:\n\n"
            "### Plan\n"
            "- A bulleted list outlining the main points of the summary.\n\n"
            "### Summary\n"
            "- A concise and well-structured summary of the topic."
        ),
        agent=planner_writer_agent,
    )

    # Create the crew with a clear process
    crew = Crew(
        agents=[planner_writer_agent],
        tasks=[high_level_task],
        process=Process.sequential,  # force sequential execution (planning then writing)
    )

    # Execute the task
    print("## Running the planning and writing task ##")
    result = crew.kickoff()

    print("\n\n---\n## Task Result ##\n---")
    print(result)

def multi_agent_collaboration():

    # Define Agents with specific roles and goals
    researcher = Agent(
        role='Senior Research Analyst',
        goal='Find and summarize the latest trends in AI.',
        backstory="You are an experienced research analyst with a knack for identifying key trends and synthesizing information.",
        verbose=True,
        llm=GEMINI_MODEL,
        allow_delegation=False,
    )

    writer = Agent(
        role='Technical Content Writer',
        goal='Write a clear and engaging blog post based on research findings.',
        backstory="You are a skilled writer who can translate complex technical topics into accessible content.",
        verbose=True,
        llm=GEMINI_MODEL,
        allow_delegation=False,
    )

    # Define Tasks for the agents
    research_task = Task(
        description="Research the top 3 emerging trends in Artificial Intelligence in 2024-2025. Focus on practical applications and potential impact.",
        expected_output="A detailed summary of the top 3 AI trends, including key points and sources.",
        agent=researcher,
    )

    writing_task = Task(
        description="Write a 500-word blog post based on the research findings. The post should be engaging and easy for a general audience to understand.",
        expected_output="A complete 500-word blog post about the latest AI trends.",
        agent=writer,
        context=[research_task],
    )

    # Create the Crew
    blog_creation_crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True # Set verbosity for detailed crew execution logs
    )

    # Execute the Crew
    print("## Running the blog creation crew with Gemini 2.0 Flash... ##")
    try:
        result = blog_creation_crew.kickoff()
        print("\n------------------\n")
        print("## Crew Final Output ##")
        print(result)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    multi_agent_collaboration()
