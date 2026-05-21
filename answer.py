import os
import logging
import time
from collections import deque
from dotenv import load_dotenv
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env if present
load_dotenv()

class GroqRateLimiter:
    """
    A simple sliding window rate limiter.
    Ensures we do not exceed max_requests within time_window seconds.
    """
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_timestamps = deque()

    def wait_if_needed(self):
        now = time.time()
        
        # Remove timestamps older than the time window
        while self.request_timestamps and now - self.request_timestamps[0] > self.time_window:
            self.request_timestamps.popleft()

        # If we have hit the maximum number of requests, we need to wait
        if len(self.request_timestamps) >= self.max_requests:
            # Sleep until the oldest request falls out of the window
            oldest = self.request_timestamps[0]
            sleep_time = self.time_window - (now - oldest)
            if sleep_time > 0:
                logger.info(f"Rate limiter active. Sleeping for {sleep_time:.2f} seconds to respect Groq {self.max_requests} RPM limit.")
                time.sleep(sleep_time)
            
            # After sleeping, pop the oldest timestamp because it's now out of the window
            self.request_timestamps.popleft()
            
            # Update 'now' after sleeping
            now = time.time()
            
        # Record the current request
        self.request_timestamps.append(now)

# Instantiate a global rate limiter for Groq: 30 requests per 60 seconds.
# We set max_requests slightly lower (e.g., 28) to be safe with latency variations.
rate_limiter = GroqRateLimiter(max_requests=28, time_window=60.0)


def _get_groq_client():
    """
    Retrieves the API key and configures the Groq client.
    Returns the configured Client instance.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY environment variable is missing or invalid. Please check your .env file.")
    
    return Groq(api_key=api_key)

# Add automatic retries for server overloads or transient 429s
@retry(
    stop=stop_after_attempt(4), 
    wait=wait_exponential(multiplier=1.5, min=2, max=10), 
    reraise=True
)
def _generate_with_retry(client, prompt: str) -> str:
    """Helper function to execute the API call with tenacity retries and rate limiting."""
    rate_limiter.wait_if_needed()
    
    logger.info("Sending prompt to Groq API (openai/gpt-oss-120b)...")
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="openai/gpt-oss-120b",
        temperature=0.2,
    )
    return chat_completion.choices[0].message.content

def generate_answer(prompt: str) -> str:
    """
    Sends a prompt to the Groq API and returns the generated answer.
    
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
        client = _get_groq_client()
        return _generate_with_retry(client, prompt)
        
    except Exception as e:
        logger.error(f"Error during Groq API generation: {e}")
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
