import os
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env if present
load_dotenv()

def _get_gemini_client():
    """
    Retrieves the API key and configures the Gemini client.
    Returns the configured Client instance.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY environment variable is missing or invalid. Please check your .env file.")
    
    # Initialize the new genai client
    return genai.Client(api_key=api_key)

# Add automatic retries for 503 server overloads
@retry(
    stop=stop_after_attempt(4), # Try up to 4 times
    wait=wait_exponential(multiplier=1.5, min=2, max=10), # Wait 2s, then 3s, then 4.5s... up to 10s
    reraise=True
)
def _generate_with_retry(client, prompt: str) -> str:
    """Helper function to execute the API call with tenacity retries."""
    logger.info("Sending prompt to Gemini API...")
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2, 
        )
    )
    return response.text

def generate_answer(prompt: str) -> str:
    """
    Sends a prompt to the Gemini API and returns the generated answer.
    
    Args:
        prompt (str): The complete prompt to send to the LLM.
        
    Returns:
        str: The generated response from the LLM.
        
    Raises:
        ValueError: If the prompt is empty.
        Exception: If the API call fails after all retries.
    """
    if not prompt or not prompt.strip():
        raise ValueError("The provided prompt is empty.")

    try:
        client = _get_gemini_client()
        return _generate_with_retry(client, prompt)
        
    except Exception as e:
        logger.error(f"Error during Gemini API generation: {e}")
        raise

if __name__ == "__main__":
    # Small test block to verify functionality directly
    print("Testing generate_answer()...")
    test_prompt = "What is the capital of France? Please answer in one short sentence."
    
    try:
        result = generate_answer(test_prompt)
        print("\n--- Model Output ---")
        print(result)
        print("--------------------")
    except Exception as err:
        print(f"\nTest failed: {err}")
