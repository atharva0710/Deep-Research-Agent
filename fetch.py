import logging
from datetime import datetime, timezone
from typing import Dict, Any

import requests
import trafilatura

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def fetch_and_extract(url: str, title: str = "No Title", domain: str = "") -> Dict[str, Any]:
    """
    Fetches a webpage and extracts clean, readable text using Trafilatura.
    
    Args:
        url (str): The URL of the webpage to fetch.
        title (str): Optional title of the webpage.
        domain (str): Optional domain of the webpage.
        
    Returns:
        Dict[str, Any]: A dictionary containing 'url', 'title', 'domain', 'extracted_text', and 'retrieved_at'.
                        If extraction fails, 'extracted_text' will be an empty string.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "url": url,
        "title": title,
        "domain": domain,
        "extracted_text": "",
        "retrieved_at": timestamp
    }
    
    if not url:
        logger.error("Provided URL is empty.")
        return result
        
    try:
        logger.info(f"Fetching URL: {url}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html_content = response.text
        
        extracted_text = trafilatura.extract(html_content)
        
        if extracted_text:
            result["extracted_text"] = extracted_text
            logger.info(f"Successfully extracted {len(extracted_text)} characters from {url}")
        else:
            logger.warning(f"Trafilatura failed to extract readable text from {url}.")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while fetching {url}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing {url}: {e}")
        
    return result

if __name__ == "__main__":
    print("Testing fetch_and_extract()...")
    test_url = "https://en.wikipedia.org/wiki/Quantum_computing"
    
    try:
        data = fetch_and_extract(test_url)
        print("\n--- Extraction Results ---")
        print(f"Timestamp: {data['retrieved_at']}")
        
        text = data['extracted_text']
        if text:
            print(f"Text Length: {len(text)} characters")
            snippet = text[:300].replace('\n', ' ')
            print(f"Snippet: {snippet}...")
        else:
            print("Failed to extract any text.")
        print("--------------------------")
    except Exception as err:
        print(f"\nTest failed: {err}")
