import sys
import time

from search import web_search
from fetch import fetch_and_extract
from answer import generate_answer
from context_builder import build_context
import memory

def print_progress(step: str):
    """Utility to print streaming-style progress updates for a better UX."""
    print(f"\n[*] {step}...")
    time.sleep(0.3)

def update_rolling_summary(session_id: str, current_summary: str, last_query: str, last_answer: str):
    """Generates an updated summary of the conversation to manage context length."""
    prompt = f"""You are maintaining a concise summary of a research conversation.
Current Summary:
{current_summary if current_summary else "No previous summary."}

New Interaction:
User: {last_query}
Agent: {last_answer}

Task: Write a highly concise updated summary capturing the main facts discussed so far. Do not exceed 2 paragraphs."""
    
    try:
        new_summary = generate_answer(prompt)
        memory.update_session_summary(session_id, new_summary)
    except Exception as e:
        print(f"[-] Warning: Failed to update summary: {e}")

import re

def generate_research_plan(session_id: str, current_query: str) -> dict:
    """
    Generates a structured research plan and standalone search query based on
    conversation history.
    """
    summary = memory.get_session_summary(session_id)
    recent_history = memory.get_session_history(session_id, limit=2)
    
    prompt = f"""You are an expert research planner and search optimizer.
Your task is to analyze the conversation history and the user's new query, and produce a research plan and standalone search query.

INSTRUCTIONS:
1. Rewrite the query to resolve all pronouns/coreferences, producing a standalone query suitable for Tavily.
2. Outline a 1-2 sentence research strategy describing what you will search for and analyze.
3. Your output must be formatted EXACTLY as follows:
QUERY: <standalone search query>
PLAN: <research strategy>

Do not include any other conversational filler, markdown formatting, or labels.

CONVERSATION SUMMARY:
{summary if summary else "None"}

RECENT HISTORY:
"""
    for turn in recent_history:
        prompt += f"User: {turn['query']}\nAgent: {turn['final_answer']}\n\n"
        
    prompt += f"USER'S NEW QUERY: {current_query}\n\nRESEARCH PLAN:"
    
    plan_obj = {
        "search_query": current_query,
        "plan_text": f"Investigating '{current_query}' directly by fetching top search results."
    }
    
    try:
        response = generate_answer(prompt).strip()
        
        # Parse the structured response
        query_match = re.search(r'QUERY:\s*(.*)', response, re.IGNORECASE)
        plan_match = re.search(r'PLAN:\s*(.*)', response, re.IGNORECASE)
        
        if query_match:
            plan_obj["search_query"] = query_match.group(1).strip()
        if plan_match:
            plan_obj["plan_text"] = plan_match.group(1).strip()
            
        return plan_obj
    except Exception as e:
        return plan_obj

def run_research_pipeline(session_id: str, question: str):
    """
    Executes the end-to-end research pipeline using planning, context builder,
    persistent memory, and a robust grounding prompt.
    """
    try:
        # Stage 1: Memory & Context
        print_progress("Loading session context")
        summary = memory.get_session_summary(session_id)
        recent_history = memory.get_session_history(session_id, limit=2)

        # Stage 1.5: Plan & Query Optimization
        print_progress("Formulating research plan")
        plan_data = generate_research_plan(session_id, question)
        search_query = plan_data["search_query"]
        research_plan = plan_data["plan_text"]
        print(f"    [Research Plan]: {research_plan}")
        print(f"    [Optimized Search Query]: '{search_query}'")

        # Stage 2: Searching the web
        print_progress("Searching web")
        search_results = web_search(query=search_query, max_results=7)
        
        opened_urls = []
        sources = []
        
        if search_results:
            print(f"    Found {len(search_results)} potential sources.")
            
            # Stage 3: Fetching sources
            print_progress("Fetching sources")
            for result in search_results:
                print(f"    -> Fetching: {result['title']}")
                # Fetch full page content and store metadata
                fetch_data = fetch_and_extract(result['url'], result['title'], result['domain'])
                
                opened_urls.append(result['url'])
                
                # Fetch metadata is already structured, add Tavily's score as well
                fetch_data["score"] = result.get("score", 0.0)
                sources.append(fetch_data)
        else:
            print("[-] No search results found. Proceeding with existing knowledge/context.")

        # Stage 4: Context Building
        print_progress("Ranking & chunking context")
        selected_chunks = build_context(question, sources, max_chars=8000)
        selected_snippets = [c['text'] for c in selected_chunks]

        # Stage 5: Generating grounded answer (via prepare_context_for_prompt)
        print_progress("Generating answer")
        
        from context_builder import prepare_context_for_prompt
        prompt = prepare_context_for_prompt(
            query=question,
            selected_chunks=selected_chunks,
            rolling_summary=summary,
            recent_history=recent_history,
            max_chars=12000
        )
        
        final_answer = generate_answer(prompt)
        
        # Stage 6: Output
        print("\n" + "="*80)
        print(" FINAL ANSWER")
        print("="*80 + "\n")
        print(final_answer)
        print("\n" + "="*80 + "\n")
        
        # Stage 7: Save to Memory
        search_queries = [search_query]
        memory.save_turn(session_id, question, search_queries, opened_urls, selected_snippets, final_answer)
        
        # Stage 8: Update Summary (Run in background conceptually, but we block here)
        print_progress("Updating session summary")
        update_rolling_summary(session_id, summary, question, final_answer)
        
        return final_answer, sources, selected_chunks

    except Exception as e:
        print(f"\n[-] Pipeline encountered an error: {e}")
        return f"Error: {str(e)}", [], []

if __name__ == "__main__":
    print("===========================================")
    print(" Minimal Deep Research Agent")
    print("===========================================")
    print("Type 'quit' or 'q' to exit.\n")
    
    # Initialize session
    session_id = memory.create_session()
    print(f"[!] Started new session: {session_id}")
    
    while True:
        try:
            user_input = input("\nEnter your research question: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nExiting...")
                break
                
            if user_input.strip():
                run_research_pipeline(session_id, user_input.strip())
            
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
