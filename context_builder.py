import re
from datetime import datetime
from typing import List, Dict, Any

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Splits text into chunks of roughly `chunk_size` characters.
    Uses a simple word-boundary split to avoid cutting words in half.
    """
    if not text:
        return []
        
    words = text.split()
    chunks = []
    current_chunk = []
    current_len = 0
    
    for word in words:
        current_chunk.append(word)
        current_len += len(word) + 1  # +1 for space
        
        if current_len >= chunk_size:
            chunks.append(" ".join(current_chunk))
            # Implement a small overlap (approx last 20% of the chunk)
            overlap_words = current_chunk[-max(1, len(current_chunk) // 5):]
            current_chunk = overlap_words
            current_len = sum(len(w) + 1 for w in current_chunk)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def _score_chunk(chunk: str, query: str, domain: str = "") -> float:
    """
    Basic relevance scoring based on term frequency overlap, 
    with a multiplier for domain reliability.
    """
    if not chunk or not query:
        return 0.0
        
    chunk_lower = chunk.lower()
    query_words = set(re.findall(r'\w+', query.lower()))
    
    # Filter common stop words for better scoring accuracy
    stop_words = {"what", "is", "the", "in", "on", "at", "to", "a", "an", "of", "and", "for", "how", "why", "are", "do", "does"}
    query_terms = query_words - stop_words
    
    if not query_terms:
        query_terms = query_words # Fallback if all words were stop words
        
    base_score = 0.0
    for term in query_terms:
        # Score based on count of exact term matches
        base_score += chunk_lower.count(term)
        
    # Domain Reliability Multiplier
    multiplier = 1.0
    domain_lower = domain.lower()
    
    reliable_suffixes = [".edu", ".gov", ".org", ".int", ".mil"]
    reliable_domains = ["nature.com", "science.org", "ncbi.nlm.nih.gov", "arxiv.org", "bbc.com", "reuters.com", "apnews.com", "npr.org", "pbs.org", "wsj.com"]
    unreliable_domains = ["quora.com", "reddit.com", "yahoo.com", "answers.com", "medium.com"]
    
    if any(domain_lower.endswith(suf) for suf in reliable_suffixes) or any(rel in domain_lower for rel in reliable_domains):
        multiplier = 2.0  # 2x boost for highly reliable sources
    elif any(unrel in domain_lower for unrel in unreliable_domains):
        multiplier = 0.5  # 50% penalty for generally unreliable sources
        
    return base_score * multiplier

def build_context(query: str, sources: List[Dict[str, Any]], max_chars: int = 8000) -> List[Dict[str, Any]]:
    """
    Chunks texts from multiple sources, scores them against the query,
    prioritizing relevance, recency, and source diversity,
    and returns top chunks within the max_chars limit.
    """
    all_chunks = []
    
    for src in sources:
        text = src.get('extracted_text') or src.get('snippet') or ""
        if not text:
            continue
            
        # Create chunks
        chunks = chunk_text(text, chunk_size=1500)
        
        # Base Tavily search score boost
        tavily_score = src.get('score', 0.0)
        
        # Recency boost (check for recent years in text or title)
        recency_boost = 0.0
        # Scan for years between 2023 and 2026
        years_in_text = re.findall(r'\b(202[3-6])\b', text)
        years_in_title = re.findall(r'\b(202[3-6])\b', src.get('title', ''))
        if years_in_title or years_in_text:
            recency_boost = 2.0  # Give a boost for fresh content containing recent years
            
        # Score each chunk
        for c in chunks:
            relevance_score = _score_chunk(c, query, src.get('domain', ''))
            # Combine scores: relevance + Tavily ranking signal + recency boost
            combined_score = relevance_score + (tavily_score * 5.0) + recency_boost
            
            all_chunks.append({
                "source_title": src['title'],
                "source_url": src['url'],
                "source_domain": src['domain'],
                "retrieved_at": src.get('retrieved_at', ''),
                "text": c,
                "score": combined_score,
                "base_relevance": relevance_score
            })
            
    # Select chunks dynamically to enforce Source Diversity
    # We sort all chunks, and then iteratively pick the best one.
    # To enforce diversity, each time we pick a chunk, we apply a decay penalty
    # to all remaining chunks that share the same URL or domain.
    selected_chunks = []
    current_len = 0
    
    # Work on a copy of chunks list so we can modify scores
    remaining_chunks = list(all_chunks)
    
    # Track the count of chunks picked from each domain to apply progressive penalty
    domain_counts = {}
    
    while remaining_chunks and current_len < max_chars:
        # Sort remaining chunks by their *penalized* score descending
        remaining_chunks.sort(key=lambda x: x['score'], reverse=True)
        
        # Pick the highest-scoring chunk
        best_chunk = remaining_chunks.pop(0)
        chunk_len = len(best_chunk['text'])
        
        if current_len + chunk_len > max_chars:
            continue
            
        # Stop selecting if scores are too low and we already have some context
        if best_chunk['score'] <= 0 and len(selected_chunks) > 0:
            break
            
        # Select this chunk
        selected_chunks.append(best_chunk)
        current_len += chunk_len
        
        # Apply diversity penalty to remaining chunks from the same domain
        domain = best_chunk['source_domain']
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Every subsequent chunk from the same domain has its score multiplied by 0.5^count
        penalty = 0.5 ** domain_counts[domain]
        for chunk in remaining_chunks:
            if chunk['source_domain'] == domain:
                chunk['score'] = chunk['score'] * penalty
                
    return selected_chunks

def prepare_context_for_prompt(
    query: str,
    selected_chunks: List[Dict[str, Any]],
    rolling_summary: str = "",
    recent_history: List[Dict[str, str]] = None,
    max_chars: int = 12000
) -> str:
    """
    Decides and structures the complete context sent to the LLM.
    Enforces a strict maximum context length (character count constraint),
    and applies a summarization fallback if the prompt exceeds limits.
    """
    if recent_history is None:
        recent_history = []

    # System instruction base
    system_base = """You are a Deep Research Agent. Your task is to answer the user's question based STRICTLY on the provided context.

REQUIREMENTS:
1. NO HALLUCINATIONS: You must base your answer ONLY on the provided context. Do not use outside knowledge.
2. EXPLICIT UNCERTAINTY & PARTIAL ANSWERS: If the context contains relevant information but cannot definitively or fully answer the exact prompt (e.g., no explicit ranking exists), DO NOT refuse to answer entirely. Instead, synthesize the information you DO have, explicitly state the limitations using phrases like "The sources do not specify" or "It is not explicitly mentioned", and provide the best possible partial answer. ONLY start your response with "I do not have enough information to answer this question." if the context is completely irrelevant or contains absolutely no helpful information.
3. CITATIONS: You MUST include inline markdown citations for every single claim. 
   - Use the EXACT format: `[Title - Domain](URL)`. 
   - Example: `The sky is blue [Space Facts - space.com](https://space.com/facts)`.
   - DO NOT use alternative brackets like 【 】 or footnotes like [1].
4. CONFLICTS: If different sources provide conflicting information, explicitly mention the conflict and cite both sources.
5. FORMAT: Use clean Markdown for readability.

"""

    def format_prompt(summary_val: str, history_val: List[Dict[str, str]], chunks_val: List[Dict[str, Any]]) -> str:
        prompt = system_base
        if summary_val:
            prompt += f"CONVERSATION SUMMARY SO FAR:\n{summary_val}\n\n"
            
        if history_val:
            prompt += "RECENT HISTORY:\n"
            for turn in history_val:
                prompt += f"User: {turn.get('query', '')}\nAgent: {turn.get('final_answer', '')}\n\n"
                
        prompt += "NEW SOURCE CONTEXT:\n"
        for i, chunk in enumerate(chunks_val):
            prompt += f"--- Source {i+1} ---\n"
            prompt += f"Title: {chunk['source_title']}\n"
            prompt += f"Domain: {chunk['source_domain']}\n"
            prompt += f"URL: {chunk['source_url']}\n"
            prompt += f"Content Snippet:\n{chunk['text']}\n\n"
            
        prompt += f"USER QUESTION: {query}\n\nANSWER:\n"
        return prompt

    # Initial try: use full history and full selected chunks
    full_prompt = format_prompt(rolling_summary, recent_history, selected_chunks)
    
    # If it fits within limit, return it
    if len(full_prompt) <= max_chars:
        return full_prompt
        
    # SUMMARIZATION FALLBACK PHASE 1: Prune older turns in recent_history, keeping the rolling summary
    # Try keeping only the very last turn of history
    pruned_history = recent_history[-1:] if recent_history else []
    full_prompt = format_prompt(rolling_summary, pruned_history, selected_chunks)
    if len(full_prompt) <= max_chars:
        return full_prompt
        
    # SUMMARIZATION FALLBACK PHASE 2: Completely remove recent_history, relying solely on rolling_summary
    full_prompt = format_prompt(rolling_summary, [], selected_chunks)
    if len(full_prompt) <= max_chars:
        return full_prompt
        
    # SUMMARIZATION FALLBACK PHASE 3: If still exceeding, iteratively prune the lowest ranking web chunks
    pruned_chunks = list(selected_chunks)
    while pruned_chunks and len(full_prompt) > max_chars:
        pruned_chunks.pop()  # Remove last (lowest scored) chunk
        full_prompt = format_prompt(rolling_summary, [], pruned_chunks)
        
    return full_prompt
