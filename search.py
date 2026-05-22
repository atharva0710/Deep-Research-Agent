import os
import logging
from typing import List, Dict, Any
from urllib.parse import urlparse
from dotenv import load_dotenv
from tavily import TavilyClient

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env if present
load_dotenv()

def _get_tavily_client() -> TavilyClient:
    """
    Retrieves the API key and initializes the TavilyClient.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key or api_key == "your_tavily_api_key_here":
        raise ValueError("TAVILY_API_KEY environment variable is missing or invalid. Please check your .env file.")
    
    return TavilyClient(api_key=api_key)

def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Performs a web search using the Tavily API and returns structured results.
    
    Args:
        query (str): The search query.
        max_results (int): Maximum number of results to return. Default is 5.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing 'title', 'url', 'domain', and 'snippet'.
    """
    if not query or not query.strip():
        raise ValueError("The search query cannot be empty.")

    try:
        client = _get_tavily_client()
        logger.info(f"Searching web for: '{query}' (max_results={max_results})")
        
        # Perform the search using the Tavily SDK
        response = client.search(
            query=query, 
            search_depth="basic",
            max_results=max_results
        )
        
        raw_results = response.get("results", [])
        structured_results = []
        
        # Extract the required fields and parse domain
        for result in raw_results:
            url = result.get("url", "")
            domain = urlparse(url).netloc if url else ""
            
            structured_results.append({
                "title": result.get("title", "No Title"),
                "url": url,
                "domain": domain,
                "snippet": result.get("content", ""),
                "score": result.get("score", 0.0)
            })
            
        logger.info(f"Successfully retrieved {len(structured_results)} results.")
        return structured_results
        
    except Exception as e:
        logger.error(f"Error during Tavily web search: {e}")
        raise

if __name__ == "__main__":
    print("Testing web_search()...")
    test_query = "What are the latest discoveries in quantum computing?"
    
    try:
        results = web_search(test_query, max_results=2)
        print(f"\n--- Search Results for '{test_query}' ---")
        for idx, res in enumerate(results, 1):
            print(f"\nResult {idx}:")
            print(f"Title:   {res['title']}")
            print(f"URL:     {res['url']}")
            print(f"Domain:  {res['domain']}")
            snippet = res['snippet'][:200].replace('\n', ' ')
            print(f"Snippet: {snippet}...")
        print("\n-----------------------------------------")
    except Exception as err:
        print(f"\nTest failed: {err}")
