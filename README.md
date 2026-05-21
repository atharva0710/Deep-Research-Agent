# Deep Research Agent (Minimal Vertical Slice)

A bare-bones implementation of a Deep Research Agent in Python, manually orchestrated without using complex frameworks like LangChain, LangGraph, or CrewAI.

## Features
- **Web Search**: Uses the Tavily API to find relevant sources.
- **Content Extraction**: Fetches webpages and extracts readable text using `trafilatura` (with `BeautifulSoup` fallback).
- **Answer Generation**: Generates comprehensive answers with citations using the Gemini API (Google AI Studio).

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**:
   Add your API keys to the `.env` file:
   ```env
   TAVILY_API_KEY="your_tavily_api_key_here"
   GEMINI_API_KEY="your_gemini_api_key_here"
   ```
   - Get a Tavily API key from [Tavily](https://tavily.com/).
   - Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/).

## Usage

Run the orchestrator script:
```bash
python app.py
```

Enter your questions at the prompt to see the agent search, extract, and answer.

## Project Structure
- `app.py`: The main orchestrator that ties everything together.
- `search.py`: Handles querying the Tavily API.
- `fetch.py`: Handles downloading HTML and extracting clean text.
- `answer.py`: Constructs the prompt with context and calls the Gemini API.
