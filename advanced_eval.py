"""
Advanced Evaluation Harness for Deep Research Agent
Implements both Heuristic and LLM-as-a-Judge metrics.
"""
import json
import re
from datetime import datetime
from google import genai
from google.genai import types
import os
import time
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from app import run_research_pipeline
import memory

load_dotenv()

def get_llm_judge():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY missing for evaluation.")
    return genai.Client(api_key=api_key)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=10, max=60),
    reraise=True
)
def llm_judge_hallucination(client, context: str, answer: str) -> float:
    """Uses LLM to detect if any claims in the answer are NOT in the context."""
    if not context.strip():
        return 0.0
        
    prompt = f"""You are an expert fact-checker. 
Read the CONTEXT and the ANSWER. 
If the ANSWER contains ANY factual claim that cannot be proven by the CONTEXT, output '0.0'. 
If all claims are fully supported by the CONTEXT, output '1.0'.
Output ONLY the number. Do not add any text.

CONTEXT:
{context}

ANSWER:
{answer}
"""
    try:
        res = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        
        # Use regex to safely extract the score in case the model adds conversational text
        match = re.search(r'(0\.0|1\.0|0|1)', res.text)
        if match:
            return float(match.group(1))
        return 0.0
    except Exception as e:
        print(f"LLM Judge Error: {e}")
        return 0.0

def evaluate_turn(turn, pipeline_output):
    """Calculates all metrics for a single turn."""
    final_answer = pipeline_output.get("answer", "")
    sources = pipeline_output.get("sources", [])
    chunks = pipeline_output.get("chunks", [])
    
    metrics = {}
    
    # 1. Retrieval Quality (Heuristic)
    authoritative_domains = [".edu", ".gov", "nature.com", "science.org", "ncbi.nlm.nih.gov"]
    metrics["authoritative_presence"] = 1.0 if any(
        any(auth in s['domain'] for auth in authoritative_domains) for s in sources
    ) else 0.0
    
    # 2. Context Selection Quality (Compression)
    total_raw_text = sum(len(s.get('extracted_text', '')) for s in sources)
    total_chunk_text = sum(len(c['text']) for c in chunks)
    metrics["compression_ratio"] = total_chunk_text / max(total_raw_text, 1)
    
    # 3. Grounding & Citations (Heuristic)
    sentences = max(len(re.split(r'[.!?]+', final_answer)), 1)
    citations = len(re.findall(r'\[.*?\]\(http.*?\)', final_answer))
    metrics["citation_density"] = min(citations / sentences, 1.0)
    
    # 4. Hallucination Check (LLM Judge)
    client = get_llm_judge()
    context_str = "\n".join([c['text'] for c in chunks])
    metrics["grounding_score"] = llm_judge_hallucination(client, context_str, final_answer)
    
    # 5. Uncertainty Handling
    if turn.get('eval_params', {}).get('must_refuse'):
        refusal_words = ["do not have enough information", "cannot answer", "not enough info", "unclear"]
        metrics["uncertainty_score"] = 1.0 if any(w in final_answer.lower() for w in refusal_words) else 0.0
    else:
        metrics["uncertainty_score"] = 1.0 # default pass if not a trap question
        
    return metrics

def run_advanced_eval():
    print("Starting Advanced LLM-as-a-Judge Evaluation...")
    with open("advanced_eval_dataset.json", "r") as f:
        dataset = json.load(f)
        
    results = []
    total_turns = 0
    
    for case in dataset["test_cases"]:
        print(f"\n======================================")
        print(f" Evaluating Case: {case['id']} ")
        print(f"======================================")
        
        session_id = memory.create_session()
        
        for turn in case["turns"]:
            print(f"\n>>> Executing Turn: {turn['query']}")
            
            # Added a 15-second pause to prevent hitting the Free Tier 15 RPM limit
            print("    [Rate Limit Protection] Pausing for 15 seconds to respect Gemini API quotas...")
            time.sleep(15)
            
            # Execute the actual pipeline
            ans, sources, chunks = run_research_pipeline(session_id, turn['query'])
            
            pipeline_output = {
                "answer": ans,
                "sources": sources,
                "chunks": chunks
            }
            
            # Evaluate the results
            metrics = evaluate_turn(turn, pipeline_output)
            results.append({
                "case_id": case['id'],
                "query": turn['query'],
                "metrics": metrics
            })
            total_turns += 1
            
    # Generate Summary Report
    print("\n\n" + "="*50)
    print(" ADVANCED EVALUATION REPORT")
    print("="*50)
    
    avg_grounding = sum(r['metrics']['grounding_score'] for r in results) / max(total_turns, 1)
    avg_compression = sum(r['metrics']['compression_ratio'] for r in results) / max(total_turns, 1)
    avg_auth = sum(r['metrics']['authoritative_presence'] for r in results) / max(total_turns, 1)
    avg_citations = sum(r['metrics']['citation_density'] for r in results) / max(total_turns, 1)
    avg_uncertainty = sum(r['metrics']['uncertainty_score'] for r in results) / max(total_turns, 1)
    
    report = {
        "Total Turns Evaluated": total_turns,
        "Avg Grounding Score (LLM Judge)": round(avg_grounding, 2),
        "Avg Context Compression Ratio": round(avg_compression, 3),
        "Avg Authoritative Presence": round(avg_auth, 2),
        "Avg Citation Density": round(avg_citations, 2),
        "Avg Uncertainty Handling": round(avg_uncertainty, 2)
    }
    
    print(json.dumps(report, indent=4))
    
    # Save to disk
    with open("advanced_eval_results.json", "w") as f:
        json.dump({"summary": report, "detailed_results": results}, f, indent=2)
    print("\n[+] Detailed results saved to advanced_eval_results.json")

if __name__ == "__main__":
    run_advanced_eval()
