import json
import os
import re
from datetime import datetime
import sys

# Suppress standard print outputs during evaluation if desired
# but let's keep them so we can see the agent working.
from app import run_research_pipeline
import memory

def load_dataset(filepath="eval_dataset.json"):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_evaluation():
    print("Starting Evaluation Harness...")
    dataset = load_dataset()
    results = []
    
    for item in dataset:
        print(f"\n{'='*50}\nEvaluating: {item['id']} ({item['type']})\n{'='*50}")
        session_id = memory.create_session()
        
        turn_answers = []
        for q in item['questions']:
            print(f"\n>> Eval Query: {q}")
            ans, _, _ = run_research_pipeline(session_id, q)
            turn_answers.append(ans)
            
        final_ans = turn_answers[-1] if turn_answers else ""
        if not final_ans:
            final_ans = ""
            
        # ----------------------------------------------------
        # HEURISTIC METRICS CALCULATION
        # ----------------------------------------------------
        
        # 1. Grounding / Usefulness: Check if expected themes are present
        themes_found = sum(1 for t in item.get('expected_themes', []) if t.lower() in final_ans.lower())
        theme_coverage = themes_found / len(item['expected_themes']) if item['expected_themes'] else 1.0
        
        # 2. Citation Integrity: Regex to find Markdown links like [Title](URL)
        citations = re.findall(r'\[.*?\]\(http.*?\)', final_ans)
        has_citations = len(citations) > 0
        
        # 3. Uncertainty Handling
        # Words the model uses when it refuses to hallucinate (expanded to support singular/plural variations)
        uncertainty_keywords = [
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
        expressed_uncertainty = any(kw in final_ans.lower() for kw in uncertainty_keywords)
        
        # If the question was explicitly an insufficient-evidence question, 
        # did the model properly state uncertainty?
        if item['type'] == 'insufficient-evidence':
            uncertainty_score = 1.0 if expressed_uncertainty else 0.0
        else:
            # If it's a factual question, it shouldn't be uncertain if the search works
            uncertainty_score = 0.0 if expressed_uncertainty else 1.0
            
        metrics = {
            "theme_coverage": theme_coverage,
            "has_citations": has_citations,
            "citation_count": len(citations),
            "expressed_uncertainty": expressed_uncertainty,
            "uncertainty_score": uncertainty_score
        }
        
        results.append({
            "id": item['id'],
            "type": item['type'],
            "questions": item['questions'],
            "expected_themes": item.get('expected_themes', []),
            "final_answer": final_ans,
            "metrics": metrics
        })
        
    # Generate summary statistics
    total = len(results)
    avg_theme_coverage = sum(r['metrics']['theme_coverage'] for r in results) / total
    pct_with_citations = sum(1 for r in results if r['metrics']['has_citations']) / total * 100
    avg_uncertainty_handling = sum(r['metrics']['uncertainty_score'] for r in results) / total
    
    summary = {
        "total_evaluations": total,
        "avg_theme_coverage": round(avg_theme_coverage, 2),
        "pct_with_citations": round(pct_with_citations, 2),
        "avg_uncertainty_score": round(avg_uncertainty_handling, 2),
        "timestamp": datetime.now().isoformat()
    }
    
    print("\n\n" + "="*50)
    print(" EVALUATION SUMMARY ")
    print("="*50)
    print(json.dumps(summary, indent=2))
    
    # Save outputs to disk
    output = {"summary": summary, "results": results}
    output_file = "eval_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[+] Detailed results saved to {output_file}")

if __name__ == "__main__":
    run_evaluation()
