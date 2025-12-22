# Copyright (c) 2025 Marco Fago
#
# This code is licensed under the MIT License.
# See the LICENSE file in the repository for the full license text.
import asyncio
import uuid

from dotenv import load_dotenv
from google.adk.agents import Agent, LlmAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

load_dotenv(".env")
# --- Define Tool Functions ---
# These functions simulate the actions of the specialist agents.
# running into 429 with gemini-2.0?
GEMINI_MODEL = "gemini-2.5-flash"
# GEMINI_MODEL = "gemini-2.5-flash"


def booking_handler(request: str) -> str:
    """
    Handles booking requests for flights and hotels.
    Args:
        request: The user's request for a booking.
    Returns:
        A confirmation message that the booking was handled.
    """
    print(
        "-------------------------- Booking Handler Called ----------------------------"
    )
    return f"Booking action for '{request}' has been simulated."


def info_handler(request: str) -> str:
    """
    Handles general information requests.
    Args:
        request: The user's question.
    Returns:
        A message indicating the information request was handled.
    """
    print("-------------------------- Info Handler Called ----------------------------")
    return (
        f"Information request for '{request}'. Result: Simulated information retrieval."
    )


def unclear_handler(request: str) -> str:
    """Handles requests that couldn't be delegated."""
    return f"Coordinator could not delegate request: '{request}'. Please clarify."


def run_agent(
    runner: InMemoryRunner, request: str, user_id: str, session_id: str
) -> list[str]:
    """
    Runs an agent pipeline and collects ALL text responses
    from all final responses across agents.
    """

    print(f"\n--- Running agent with request: '{request}' ---")

    responses: list[str] = []

    try:
        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=request)],
            ),
        ):

            if not (event.is_final_response() and event.content):
                continue

            text = None

            if getattr(event.content, "text", None):
                text = event.content.text
            elif event.content.parts:
                text = event.author +":\n"+ "".join(part.text for part in event.content.parts if part.text)

            if text:
                responses.append(text)

        print("Collected responses:")
        for r in responses:
            print("-", r)

        return responses

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


async def new_session(coordinator: BaseAgent):
    runner = InMemoryRunner(coordinator)
    user_id = "user_123"
    session_id = str(uuid.uuid4())
    await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )

    return runner, user_id, session_id


async def routing():
    """Main function to run the ADK example."""
    print("--- Google ADK Routing Example (ADK Auto-Flow Style) ---")
    print("Note: This requires Google ADK installed and authenticated.")

    # --- Create Tools from Functions ---
    booking_tool = FunctionTool(booking_handler)
    info_tool = FunctionTool(info_handler)

    # Define specialized sub-agents equipped with their respective tools
    booking_agent = Agent(
        name="Booker",
        model=GEMINI_MODEL,
        description="A specialized agent that handles all flight and hotel booking requests by calling the booking tool.",
        tools=[booking_tool],
    )

    info_agent = Agent(
        name="Info",
        model=GEMINI_MODEL,
        description="A specialized agent that provides general information and answers user questions by calling the info tool.",
        tools=[info_tool],
    )

    # Define the parent agent with explicit delegation instructions
    coordinator = Agent(
        name="Coordinator",
        model=GEMINI_MODEL,
        instruction=(
            "You are the main coordinator. Your only task is to analyze incoming user requests "
            "and delegate them to the appropriate specialist agent. Do not try to answer the user directly.\n"
            "- For any requests related to booking flights or hotels, delegate to the 'Booker' agent.\n"
            "- For all other general information questions, delegate to the 'Info' agent."
        ),
        description="A coordinator that routes user requests to the correct specialist agent.",
        # The presence of sub_agents enables LLM-driven delegation (Auto-Flow) by default.
        sub_agents=[booking_agent, info_agent],
    )

    runner, user_id, session_id = await new_session(coordinator)
    # Example Usage
    result_b = run_agent(
        runner, "What is the highest mountain in the world?", user_id, session_id
    )
    result_a = run_agent(runner, "Book me a hotel in Paris.", user_id, session_id)
    print(f"Final Output A: {result_a}")
    print(f"Final Output B: {result_b}")
    result_c = run_agent(
        runner, "Tell me a random fact.", user_id, session_id
    )  # Should go to Info
    print(f"Final Output C: {result_c}")
    result_d = run_agent(
        runner, "Find flights to Tokyo next month.", user_id, session_id
    )  # Should go to Booker
    print(f"Final Output D: {result_d}")


def google_search(request: str) -> str:
    """
    Handles searching the net for information
    Args:
        request: The user's request for a booking.
    Returns:
        A confirmation message that the search was handled.
    """
    print(
        "-------------------------- Google Search Called ----------------------------"
    )
    return f"Search action for '{request}' has been simulated."


async def parallel_execution():
    # Part of agent.py --> Follow https://google.github.io/adk-docs/get-started/quickstart/ to learn the setup

    # Researcher 1: Renewable Energy
    researcher_agent_1 = LlmAgent(
        name="RenewableEnergyResearcher",
        model=GEMINI_MODEL,
        instruction="""You are an AI Research Assistant specializing in energy.
    Research the latest advancements in 'renewable energy sources'.
    Use the Google Search tool provided.
    Summarize your key findings concisely (1-2 sentences).
    Output *only* the summary.
    """,
        description="Researches renewable energy sources.",
        tools=[google_search],
        # Store result in state for the merger agent
        output_key="renewable_energy_result",
    )

    # Researcher 2: Electric Vehicles
    researcher_agent_2 = LlmAgent(
        name="EVResearcher",
        model=GEMINI_MODEL,
        instruction="""You are an AI Research Assistant specializing in transportation.
    Research the latest developments in 'electric vehicle technology'.
    Use the Google Search tool provided.
    Summarize your key findings concisely (1-2 sentences).
    Output *only* the summary.
    """,
        description="Researches electric vehicle technology.",
        tools=[google_search],
        # Store result in state for the merger agent
        output_key="ev_technology_result",
    )

    # Researcher 3: Carbon Capture
    researcher_agent_3 = LlmAgent(
        name="CarbonCaptureResearcher",
        model=GEMINI_MODEL,
        instruction="""You are an AI Research Assistant specializing in climate solutions.
    Research the current state of 'carbon capture methods'.
    Use the Google Search tool provided.
    Summarize your key findings concisely (1-2 sentences).
    Output *only* the summary.
    """,
        description="Researches carbon capture methods.",
        tools=[google_search],
        # Store result in state for the merger agent
        output_key="carbon_capture_result",
    )

    # --- 2. Create the ParallelAgent (Runs researchers concurrently) ---
    # This agent orchestrates the concurrent execution of the researchers.
    # It finishes once all researchers have completed and stored their results in state.
    parallel_research_agent = ParallelAgent(
        name="ParallelWebResearchAgent",
        sub_agents=[researcher_agent_1, researcher_agent_2, researcher_agent_3],
        description="Runs multiple research agents in parallel to gather information.",
    )

    # --- 3. Define the Merger Agent (Runs *after* the parallel agents) ---
    # This agent takes the results stored in the session state by the parallel agents
    # and synthesizes them into a single, structured response with attributions.
    merger_agent = LlmAgent(
        name="SynthesisAgent",
        model=GEMINI_MODEL,  # Or potentially a more powerful model if needed for synthesis
        instruction="""You are an AI Assistant responsible for combining research findings into a structured report.

    Your primary task is to synthesize the following research summaries, clearly attributing findings to their source areas. Structure your response using headings for each topic. Ensure the report is coherent and integrates the key points smoothly.

    **Crucially: Your entire response MUST be grounded *exclusively* on the information provided in the 'Input Summaries' below. Do NOT add any external knowledge, facts, or details not present in these specific summaries.**

    **Input Summaries:**

    *   **Renewable Energy:**
        {renewable_energy_result}

    *   **Electric Vehicles:**
        {ev_technology_result}

    *   **Carbon Capture:**
        {carbon_capture_result}

    **Output Format:**

    ## Summary of Recent Sustainable Technology Advancements

    ### Renewable Energy Findings
    (Based on RenewableEnergyResearcher's findings)
    [Synthesize and elaborate *only* on the renewable energy input summary provided above.]

    ### Electric Vehicle Findings
    (Based on EVResearcher's findings)
    [Synthesize and elaborate *only* on the EV input summary provided above.]

    ### Carbon Capture Findings
    (Based on CarbonCaptureResearcher's findings)
    [Synthesize and elaborate *only* on the carbon capture input summary provided above.]

    ### Overall Conclusion
    [Provide a brief (1-2 sentence) concluding statement that connects *only* the findings presented above.]

    Output *only* the structured report following this format. Do not include introductory or concluding phrases outside this structure, and strictly adhere to using only the provided input summary content.
    """,
        description="Combines research findings from parallel agents into a structured, cited report, strictly grounded on provided inputs.",
        # No tools needed for merging
        # No output_key needed here, as its direct response is the final output of the sequence
    )

    # --- 4. Create the SequentialAgent (Orchestrates the overall flow) ---
    # This is the main agent that will be run. It first executes the ParallelAgent
    # to populate the state, and then executes the MergerAgent to produce the final output.
    sequential_pipeline_agent = SequentialAgent(
        name="ResearchAndSynthesisPipeline",
        # Run parallel research first, then merge
        sub_agents=[parallel_research_agent, merger_agent],
        description="Coordinates parallel research and synthesizes the results.",
    )

    root_agent = sequential_pipeline_agent
    runner, user_id, session_id = await new_session(root_agent)
    run_agent(
        runner,
        "Conduct research on recent advancements in sustainable technologies and provide a structured report.",
        user_id,
        session_id,
    )


async def reflection():
    # The first agent generates the initial draft.
    generator = LlmAgent(
        name="DraftWriter",
        description="Generates initial draft content on a given subject.",
        instruction="Write a short, informative paragraph about the user's subject.",
        output_key="draft_text",  # The output is saved to this state key for intermediate processing. 
        # it gets written out to event.actions.state_delta
        model=GEMINI_MODEL,
    )

    # The second agent critiques the draft from the first agent.
    reviewer = LlmAgent(
        name="FactChecker",
        description="Reviews a given text for factual accuracy and provides a structured critique.",
        instruction="""
        You are a meticulous fact-checker.
        1. Read the text provided in the state key 'draft_text'.
        2. Carefully verify the factual accuracy of all claims.
        3. Your final output must be a dictionary containing two keys:
        - "status": A string, either "ACCURATE" or "INACCURATE".
        - "reasoning": A string providing a clear explanation for your status, citing specific issues if any are found.
        """,
        output_key="review_output",  # The structured dictionary is saved here.
        model=GEMINI_MODEL,
    )

    # The SequentialAgent ensures the generator runs before the reviewer.
    review_pipeline = SequentialAgent(
        name="WriteAndReview_Pipeline", sub_agents=[generator, reviewer]
    )
    runner, user_id, session_id = await new_session(review_pipeline)
    run_agent(
        runner,
        "The impact of climate change on polar bear populations.",
        user_id=user_id,
        session_id=session_id,
    )

    # Execution Flow:
    # 1. generator runs -> saves its paragraph to state['draft_text'].
    # 2. reviewer runs -> reads state['draft_text'] and saves its dictionary output to state['review_output'].
        # the output of the reviewer is a code box with a json inside
        # should it be parsed? with pydantic?


async def main():
    await reflection()


if __name__ == "__main__":
    asyncio.run(main())
