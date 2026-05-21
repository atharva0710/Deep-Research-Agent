import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env if present
load_dotenv()

def _get_gemini_model():
    """
    Retrieves the API key and configures the Gemini client.
    Returns the configured GenerativeModel instance.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY environment variable is missing or invalid. Please check your .env file.")
    
    genai.configure(api_key=api_key)
    
    # Initialize the model with gemini-1.5-flash as requested
    return genai.GenerativeModel('gemini-1.5-flash')

def generate_answer(prompt: str) -> str:
    """
    Sends a prompt to the Gemini API and returns the generated answer.
    
    Args:
        prompt (str): The complete prompt to send to the LLM.
        
    Returns:
        str: The generated response from the LLM.
        
    Raises:
        ValueError: If the prompt is empty.
        Exception: If the API call fails.
    """
    if not prompt or not prompt.strip():
        raise ValueError("The provided prompt is empty.")

    try:
        model = _get_gemini_model()
        
        # Configure generation parameters to ensure deterministic, factual answers
        generation_config = genai.GenerationConfig(
            temperature=0.2, 
        )
        
        logger.info("Sending prompt to Gemini API...")
        response = model.generate_content(prompt, generation_config=generation_config)
        
        return response.text
        
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
