# Deep Research Agent

## Project Overview
The Deep Research Agent is a minimal, end-to-end Python application designed to autonomously research complex queries by synthesizing real-time data from the web. It is constructed without relying on heavy orchestration frameworks (such as LangChain or CrewAI), ensuring maximum transparency and control over the execution pipeline. The agent fetches multiple web sources, dynamically builds and ranks context chunks, manages stateful conversations across sessions, and grounds its final Large Language Model (LLM) responses with explicit source citations.

## Architecture
The system is divided into modular, highly decoupled components:
- **Orchestrator (`app.py`)**: The central pipeline managing state transitions (planning, searching, fetching, chunking, and generation).
- **Search Engine (`search.py`)**: Interfaces with the Tavily API to dynamically retrieve a curated list of relevant URLs and snippets based on the user's query.
- **Content Extractor (`fetch.py`)**: Downloads web pages and parses readable article text while stripping HTML boilerplate (using Trafilatura and Requests).
- **Context Builder (`context_builder.py`)**: Chunks large documents, applies a lightweight TF-based relevance scoring algorithm, and truncates the context payload to optimize the LLM's context window.
- **Answer Generator (`answer.py`)**: Structures the final prompt enforcing strict citation rules and uncertainty statements, interacting with Google's Gemini 1.5 Flash.
- **Session Memory (`memory.py`)**: Provides a persistent SQLite-backed storage system tracking conversational turns, search metrics, and generating rolling summaries for multi-turn dependency management.
- **Evaluation Harness (`eval.py`)**: An automated testing script measuring grounding quality, citation integrity, and uncertainty handling against a predefined dataset.

## Setup Instructions
1. Clone the repository to your local machine.
2. Ensure you have Python 3.9 or newer installed.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Variables
Create a `.env` file in the root directory and populate it with the required API keys. Do not commit this file to version control.
```env
TAVILY_API_KEY="your_tavily_api_key_here"
GEMINI_API_KEY="your_gemini_api_key_here"
```
- A Tavily API key is required for automated web search capabilities.
- A Google Gemini API key (via Google AI Studio) is required for answer synthesis and rolling summaries.

## How to Run

To start the interactive research pipeline:
```bash
python app.py
```
Type your query when prompted. Type `quit` or `exit` to terminate the session.

To run the automated evaluation harness:
```bash
python eval.py
```

## Design Note
A primary design principle of this project is the exclusion of heavy LLM orchestrator frameworks. By writing the retrieval, parsing, chunking, and prompt-generation logic manually, this architecture minimizes hidden abstractions. This guarantees that latency, token consumption, context formatting, and error-handling mechanisms are completely visible and adjustable without framework overhead.

## Evaluation Methodology
The project includes a custom evaluation harness (`eval.py`) paired with an evaluation dataset (`eval_dataset.json`). This system heuristically assesses:
- **Theme Coverage**: Validates if the agent's output contains the necessary factual themes required by the evaluation benchmark.
- **Citation Integrity**: Verifies that the LLM successfully incorporates Markdown-formatted inline citations.
- **Uncertainty Handling**: Tests the agent against unanswerable queries (insufficient evidence) to confirm the model declines to answer rather than hallucinating facts.

## Limitations
- **Naive Chunking**: The current text chunking logic relies on string manipulation. It does not utilize semantic tokenizers (e.g., Tiktoken) or dense vector embeddings for retrieval, which could limit context precision.
- **Asynchronous Execution**: Network operations (fetching multiple sources) are currently synchronous.
- **Static Search Depth**: The Tavily API query parameters are static; a more sophisticated agent would dynamically adjust search depth and rewrite queries iteratively.

## Future Improvements
- **Semantic Routing & RAG**: Upgrade `context_builder.py` to leverage an embedding model and a vector database for true Semantic Search rather than frequency-based scoring.
- **Parallel Network Requests**: Refactor `fetch.py` to utilize `asyncio` and `aiohttp` to fetch multiple URLs concurrently.
- **LLM-as-a-Judge Evaluation**: Expand the evaluation harness to utilize a larger parameter model (e.g., Gemini 1.5 Pro) to grade the final answers on logical consistency and nuance.
- **Frontend UI**: Transition from a terminal-based CLI to a web interface using Streamlit or Gradio.

## Example Conversations

**User**: What are the main differences between the iPhone 14 Pro and iPhone 15 Pro?
**Agent**: 
*... (Streaming updates: Loading session context, Searching web, Fetching sources, Ranking context, Generating answer) ...*

Based on the sources analyzed, the primary differences between the iPhone 14 Pro and iPhone 15 Pro include:
1. Material: The iPhone 15 Pro transitions from the stainless steel frame of the 14 Pro to an aerospace-grade titanium design, making it noticeably lighter [iPhone 15 Pro Review — The Verge](https://www.theverge.com/...).
2. Connector: The iPhone 15 Pro replaces the proprietary Lightning port with a standard USB-C port.
3. Processor: The 15 Pro utilizes the A17 Pro chip compared to the A16 Bionic in the 14 Pro.

**User**: What exact brand of socks did Albert Einstein wear on his 50th birthday?
**Agent**: 
*... (Streaming updates) ...*

I do not have enough information to answer this question. The provided sources discuss Albert Einstein's life and scientific achievements but do not contain explicit evidence regarding the specific brand of socks he wore on his 50th birthday.
