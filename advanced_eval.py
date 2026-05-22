"""
Advanced Evaluation Harness for Deep Research Agent
Implements both Heuristic and LLM-as-a-Judge metrics.
"""
import json
import re
from datetime import datetime
from groq import Groq
import os
import time
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from app import run_research_pipeline
import memory

load_dotenv()

def get_llm_judge():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing for evaluation.")
    return Groq(api_key=api_key)

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
        from answer import rate_limiter
        rate_limiter.wait_if_needed()
        
        res = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        text = res.choices[0].message.content
        match = re.search(r'(0\.0|1\.0|0|1)', text)
        if match:
            return float(match.group(1))
        return 0.0
    except Exception as e:
        print(f"LLM Judge Grounding Error: {e}")
        return 0.0

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=10, max=60),
    reraise=True
)
def llm_judge_conflict_resolution(client, answer: str) -> float:
    """Uses LLM to evaluate if the answer successfully flags conflicting viewpoints/studies."""
    prompt = f"""You are an expert evaluator. 
Read the provided ANSWER. 
Determine if the ANSWER explicitly mentions conflicting viewpoints, divergent findings, or disagreement among studies/sources.
If the ANSWER identifies the conflict and cites multiple sides/perspectives, output '1.0'.
Otherwise, output '0.0'.
Output ONLY the number. Do not add any text.

ANSWER:
{answer}
"""
    try:
        from answer import rate_limiter
        rate_limiter.wait_if_needed()
        
        res = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        text = res.choices[0].message.content
        match = re.search(r'(0\.0|1\.0|0|1)', text)
        if match:
            return float(match.group(1))
        return 0.0
    except Exception as e:
        print(f"LLM Judge Conflict Error: {e}")
        return 0.0

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=10, max=60),
    reraise=True
)
def llm_judge_memory_continuity(client, query: str, answer: str) -> float:
    """Uses LLM to evaluate if the answer successfully resolves coreferences and maintains context from previous turns."""
    prompt = f"""You are an expert evaluator. 
Read the user's query: "{query}" and the agent's ANSWER.
Determine if the agent successfully remembered the context of the previous conversation (e.g. which electric vehicles were discussed) to answer the query correctly.
If the ANSWER correctly addresses the query using the conversational context, output '1.0'.
Otherwise, output '0.0'.
Output ONLY the number. Do not add any text.

QUERY: {query}
ANSWER:
{answer}
"""
    try:
        from answer import rate_limiter
        rate_limiter.wait_if_needed()
        
        res = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        text = res.choices[0].message.content
        match = re.search(r'(0\.0|1\.0|0|1)', text)
        if match:
            return float(match.group(1))
        return 0.0
    except Exception as e:
        print(f"LLM Judge Memory Error: {e}")
        return 0.0

def evaluate_turn(turn, pipeline_output):
    """Calculates all metrics for a single turn."""
    final_answer = pipeline_output.get("answer", "")
    sources = pipeline_output.get("sources", [])
    chunks = pipeline_output.get("chunks", [])
    eval_params = turn.get('eval_params', {})
    
    metrics = {}
    client = get_llm_judge()
    
    # 1. Retrieval Quality (Heuristic)
    authoritative_domains = [".edu", ".gov", "nature.com", "science.org", "ncbi.nlm.nih.gov"]
    metrics["authoritative_presence"] = 1.0 if any(
        any(auth in s['domain'] for auth in authoritative_domains) for s in sources
    ) else 0.0
    
    # 2. Context Selection Quality (Compression)
    total_raw_text = sum(len(s.get('extracted_text') or s.get('snippet') or '') for s in sources)
    total_chunk_text = sum(len(c['text']) for c in chunks)
    metrics["compression_ratio"] = total_chunk_text / max(total_raw_text, 1)
    
    # 3. Grounding & Citations (Heuristic)
    text_no_urls = re.sub(r'http[s]?://\S+', '', final_answer)
    sentences = max(len(re.split(r'[.!?]+(?:\s|$)', text_no_urls)), 1)
    citations = len(re.findall(r'(?:\[.*?\]\(http.*?\))|(?:【.*?】)', final_answer))
    metrics["citation_density"] = min(citations / sentences, 1.0)
    
    # 4. Hallucination Check (LLM Judge)
    context_str = "\n".join([c['text'] for c in chunks])
    metrics["grounding_score"] = llm_judge_hallucination(client, context_str, final_answer)
    
    # 5. Uncertainty Handling
    if eval_params.get('must_refuse'):
        refusal_words = [
            "do not have enough information", 
            "does not contain enough information", 
            "does not contain information", 
            "does not specify", 
            "not explicitly mentioned", 
            "does not mention", 
            "not enough information", 
            "not enough info", 
            "cannot answer", 
            "cannot be answered", 
            "not possible to determine", 
            "insufficient evidence",
            "not explicitly stated"
        ]
        metrics["uncertainty_score"] = 1.0 if any(w in final_answer.lower() for w in refusal_words) else 0.0
    else:
        metrics["uncertainty_score"] = 1.0
        
    # 6. Source Diversity (Heuristic)
    unique_domains = set(c['source_domain'] for c in chunks)
    if len(unique_domains) >= 3:
        metrics["source_diversity"] = 1.0
    elif len(unique_domains) == 2:
        metrics["source_diversity"] = 0.7
    elif len(unique_domains) == 1:
        metrics["source_diversity"] = 0.4
    else:
        metrics["source_diversity"] = 0.0

    # 7. Conflict Resolution (LLM Judge)
    if eval_params.get("must_flag_conflict"):
        metrics["conflict_score"] = llm_judge_conflict_resolution(client, final_answer)
    else:
        metrics["conflict_score"] = None

    # 8. Memory Continuity (LLM Judge)
    if eval_params.get("requires_memory_continuity"):
        metrics["memory_score"] = llm_judge_memory_continuity(client, turn["query"], final_answer)
    else:
        metrics["memory_score"] = None
        
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
            
            # Execute the pipeline
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
    avg_diversity = sum(r['metrics']['source_diversity'] for r in results) / max(total_turns, 1)
    
    conflict_cases = [r['metrics']['conflict_score'] for r in results if r['metrics']['conflict_score'] is not None]
    avg_conflict = sum(conflict_cases) / len(conflict_cases) if conflict_cases else 1.0
    
    memory_cases = [r['metrics']['memory_score'] for r in results if r['metrics']['memory_score'] is not None]
    avg_memory = sum(memory_cases) / len(memory_cases) if memory_cases else 1.0
    
    report = {
        "Total Turns Evaluated": total_turns,
        "Avg Grounding Score (LLM Judge)": round(avg_grounding, 2),
        "Avg Context Compression Ratio": round(avg_compression, 3),
        "Avg Authoritative Presence": round(avg_auth, 2),
        "Avg Citation Density": round(avg_citations, 2),
        "Avg Uncertainty Handling": round(avg_uncertainty, 2),
        "Avg Source Diversity": round(avg_diversity, 2),
        "Avg Conflict Resolution Score (LLM Judge)": round(avg_conflict, 2),
        "Avg Memory Continuity Score (LLM Judge)": round(avg_memory, 2)
    }
    
    print(json.dumps(report, indent=4))
    
    # Save to disk
    with open("advanced_eval_results.json", "w") as f:
        json.dump({"summary": report, "detailed_results": results}, f, indent=2)
    print("\n[+] Detailed results saved to advanced_eval_results.json")

if __name__ == "__main__":
    run_advanced_eval()
