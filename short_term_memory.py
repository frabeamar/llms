from typing import Any

import typer
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel

from model import load_gemini_chat

app = typer.Typer()


def get_user_info(name: str) -> str:
    """Get user information."""
    return f"User name is {name}."


chatbot = ChatOllama(model="llama3.1")


class CustomAgentState(AgentState):
    user_id: str
    preferences: dict


@app.command()
def short_memory_with_db():
    # save everything into a database
    DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()  # auto create tables in PostgresSql
        agent = create_agent(
            chatbot,  # or another available model like "gpt-4o"
            [get_user_info],
            checkpointer=checkpointer,
            state_schema=CustomAgentState,
        )
        # agent.invoke(
        #     {"messages": [{"role": "user", "content": "Hi! My name is Alice."}]},
        #     {"configurable": {"thread_id": "2"}},
        # )
        new_msg = agent.invoke(
            {
                "messages": [{"role": "user", "content": "My name is alice"}],
                "user_id": "user_123",
                "preferences": {"theme": "dark"},
            },
            {"configurable": {"thread_id": "2"}},
        )
        print(new_msg["messages"][-1].content)
        # cannot differentiate between users?
        # wrong setup?
        # when does data get stale?
        # what do you need the thread for?


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Trim messages to the last 3."""
    messages = state["messages"]
    if len(messages) <= 3:
        return None
    trimmed_messages = messages[-3:]
    return {"messages": [RemoveMessage(REMOVE_ALL_MESSAGES), *trimmed_messages]}


@app.command()
def trimmed_memory():
    agent = create_agent(
        chatbot, middleware=[trim_messages], checkpointer=InMemorySaver()
    )
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    agent.invoke({"messages": "hi, my name is bob"}, config)
    agent.invoke({"messages": "write a short poem about cats"}, config)
    agent.invoke({"messages": "now do the same but for dogs"}, config)
    final_response = agent.invoke({"messages": "what's my name?"}, config)

    final_response["messages"][-1].pretty_print()

    agent.invoke({"messages": "hi, my name is bob"}, config)
    final_response = agent.invoke({"messages": "what's my name?"}, config)
    final_response["messages"][-1].pretty_print()


class CustomState(AgentState):
    user_name: str


class CustomContext(BaseModel):
    user_id: str


@tool
def update_user_info(runtime: ToolRuntime[CustomContext, CustomAgentState]):
    """Get user information."""
    user_id = runtime.context.user_id
    name = "Alice" if user_id == "user_1" else "unknown"
    print("update", name)
    print(runtime.state)
    return Command(
        update={
            "user_name": name,
            "messages": [
                ToolMessage(
                    "Successfully looked up user information",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )


@tool
def greet(runtime: ToolRuntime[CustomContext, CustomState]):
    """Use this to greet the user once you found their info."""
    user_name = runtime.state.get("user_name", None)
    print("username", user_name)
    print(runtime.state)
    if user_name is None:
        return Command(
        update={
            "messages": [
                ToolMessage(
                    "Please call the 'update_user_info' tool it will get and update the user's name.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
     
        }
    )
    print("greet",  user_name)

    return f"Hello {user_name}!"



@app.command()
def modify_short_term_memory():
    agent = create_agent(
        load_gemini_chat(),
        context_schema=CustomContext,
        state_schema=CustomState,
        tools=[update_user_info, greet],
    )

    res = agent.invoke(
        {"messages": [{"role": "user", "content": "greet the user with their name"}]},
        context=CustomContext(user_id="user_1"),
    )
    for m in res["messages"]:
        print(m.content[0].text)

    agent.invoke(
        {"messages": [{"role": "user", "content": "greet the user"}]},
        context=CustomContext(user_id="user_2"),
    )
    for m in res["messages"]:
        print(m.content)


app()
