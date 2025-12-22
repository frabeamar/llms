# experiment with LLMs
Here are some projects I completed to understand the world of llms better

## Environment. 
I am using uv to run python. Simply run the file with 
```
uv run <filename> <entrypoint>
```
All the dependency in pyproject.toml will be automatically installed
My api keys are saved into a `.env` file. touch and copy your own to run the gemini model

## 1) chat with a pdf
I am using an opensource model [Ollama](https://ollama.com/). 
To download a model first download and install the ollama server
```
 curl -fsSL https://ollama.com/install.sh | sh
```
see what models are available
```
ollama list
```
and pull the one you need
```
ollama pull llama3.1
```


I am saving all the information from the pdf in a vector database and then I can use the llm model to ask question about the file!


## google adk
The `google_adk.py` script demonstrates several patterns for building multi-agent systems using the Google Agent Development Kit (ADK). It showcases agent orchestration for routing, parallel execution, reflection, and the use of built-in tools.

#### Key Patterns and Features:

1.  **Agent Routing (Auto-Flow)**
    -   The `routing()` function shows a `Coordinator` agent that intelligently delegates tasks to specialized sub-agents (`Booker` and `Info`). This demonstrates LLM-driven delegation based on agent descriptions and instructions.

2.  **Parallel Execution & Synthesis**
    -   The `parallel_execution()` function implements a fan-out/fan-in workflow.
    -   A `ParallelAgent` runs multiple researcher agents concurrently.
    -   A `SequentialAgent` then passes their collective outputs (stored in the session state via `output_key`) to a `SynthesisAgent` which combines the information into a final, structured report.

3.  **Reflection (Generate & Review)**
    -   The `reflection()` function demonstrates a two-step "reflection" loop.
    -   A `DraftWriter` agent generates content.
    -   A `FactChecker` agent then reviews the generated content for accuracy.
    -   This sequence is managed by a `SequentialAgent`.

4.  **Using Built-in Tools & Code Execution**
    -   **Google Search**: The `builtin_google_search()` function shows a simple agent using the pre-built `google_search` tool.
    -   **Code Execution**: The `call_agent_async()` function features an agent using the `BuiltInCodeExecutor` to perform calculations by writing and running Python code. It also illustrates how to process the asynchronous event stream to inspect the generated code and its execution results.
