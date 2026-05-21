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

def run_research_pipeline(session_id: str, question: str):
    """
    Executes the minimal end-to-end research pipeline using context builder,
    persistent memory, and a robust grounding prompt.
    """
    try:
        # Stage 1: Memory & Context
        print_progress("Loading session context")
        summary = memory.get_session_summary(session_id)
        recent_history = memory.get_session_history(session_id, limit=2)

        # Stage 2: Searching the web
        print_progress("Searching web")
        search_results = web_search(query=question, max_results=3)
        
        opened_urls = []
        sources = []
        
        if search_results:
            print(f"    Found {len(search_results)} potential sources.")
            
            # Stage 3: Fetching sources
            print_progress("Fetching sources")
            for result in search_results:
                print(f"    -> Fetching: {result['title']}")
                fetch_data = fetch_and_extract(result['url'])
                
                opened_urls.append(result['url'])
                
                source_obj = {
                    "title": result['title'],
                    "url": result['url'],
                    "domain": result['domain'],
                    "snippet": result['snippet'],
                    "extracted_text": fetch_data.get('extracted_text', ''),
                    "retrieved_at": fetch_data.get('retrieved_at', '')
                }
                sources.append(source_obj)
        else:
            print("[-] No search results found. Proceeding with existing knowledge/context.")

        # Stage 4: Context Building
        print_progress("Ranking & chunking context")
        selected_chunks = build_context(question, sources, max_chars=8000)
        selected_snippets = [c['text'] for c in selected_chunks]

        # Stage 5: Generating grounded answer
        print_progress("Generating answer")
        
        prompt = f"""You are a Deep Research Agent. Your task is to answer the user's question based STRICTLY on the provided context.

REQUIREMENTS:
1. NO HALLUCINATIONS: You must base your answer ONLY on the provided context. Do not use outside knowledge.
2. EXPLICIT UNCERTAINTY: If the context does not contain enough evidence to fully answer the question, state that clearly. If the evidence is weak, note that it is weak.
3. CITATIONS: You MUST include inline citations for every claim. Use the exact format: [Title — Domain](URL).
4. CONFLICTS: If different sources provide conflicting information, explicitly mention the conflict.
5. FORMAT: Use clean Markdown for readability.

"""
        if summary:
            prompt += f"CONVERSATION SUMMARY SO FAR:\n{summary}\n\n"
            
        if recent_history:
            prompt += "RECENT HISTORY:\n"
            for turn in recent_history:
                prompt += f"User: {turn['query']}\nAgent: {turn['final_answer']}\n\n"

        prompt += "NEW SOURCE CONTEXT:\n"
        for i, chunk in enumerate(selected_chunks):
            prompt += f"--- Source {i+1} ---\n"
            prompt += f"Title: {chunk['source_title']}\n"
            prompt += f"Domain: {chunk['source_domain']}\n"
            prompt += f"URL: {chunk['source_url']}\n"
            prompt += f"Content Snippet:\n{chunk['text']}\n\n"

        prompt += f"USER QUESTION: {question}\n\n"
        prompt += "ANSWER:\n"
        
        final_answer = generate_answer(prompt)
        
        # Stage 6: Output
        print("\n" + "="*80)
        print(" FINAL ANSWER")
        print("="*80 + "\n")
        print(final_answer)
        print("\n" + "="*80 + "\n")
        
        # Stage 7: Save to Memory
        search_queries = [question] # For now, query is just user's question
        memory.save_turn(session_id, question, search_queries, opened_urls, selected_snippets, final_answer)
        
        # Stage 8: Update Summary (Run in background conceptually, but we block here)
        print_progress("Updating session summary")
        update_rolling_summary(session_id, summary, question, final_answer)
        
        return final_answer

    except Exception as e:
        print(f"\n[-] Pipeline encountered an error: {e}")
        return f"Error: {str(e)}"

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
