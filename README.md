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

