from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.config import get_stream_writer

# @tool missing since the function is so easy
def get_weather(city: str) -> str:
    """Get the weather in a given city."""
    return f"It is sunny in {city}."


def get_weather_custom_updates(city: str) -> str:
    """Get the weather in a given city."""
    writer = get_stream_writer()
    writer("Fetching weather data...\n")
    return f"It is sunny in {city}."
chatbot = ChatOllama(model="llama3.1")

agent = create_agent(
    chatbot,
    tools=[get_weather],
)

message = {"messages": [{"role": "user", "content": "What's the weather in New York?"}]}
for step in agent.stream(
    message,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

for chunk in agent.stream(
    message,
    stream_mode="updates",
):
    for step, data in chunk.items():
        print(f"Update for step {step}: {data['messages'][-1].content_blocks}")

for token, metadata in agent.stream(
    message,
    stream_mode="messages",
):
    print(f"node: {metadata['langgraph_node']}")
    print(f"content: {token.content_blocks}")
    print("\n")

agent = create_agent(chatbot, tools=[get_weather_custom_updates])
for step in agent.stream(
    message,
    stream_mode="custom",
):
    # the return never gets printed here only the updates
    print(step)
