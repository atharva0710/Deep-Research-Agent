import re
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

def _score_chunk(chunk: str, query: str) -> float:
    """
    Basic relevance scoring based on term frequency overlap.
    A lightweight alternative to embeddings for simple pipelines.
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
        
    score = 0.0
    for term in query_terms:
        # Score based on count of exact term matches
        score += chunk_lower.count(term)
        
    return score

def build_context(query: str, sources: List[Dict[str, Any]], max_chars: int = 8000) -> List[Dict[str, Any]]:
    """
    Chunks texts from multiple sources, scores them against the query, 
    and returns top chunks within the max_chars limit.
    """
    all_chunks = []
    
    for src in sources:
        text = src.get('extracted_text') or src.get('snippet') or ""
        if not text:
            continue
            
        # Create chunks
        chunks = chunk_text(text, chunk_size=1500)
        
        # Score each chunk
        for c in chunks:
            score = _score_chunk(c, query)
            all_chunks.append({
                "source_title": src['title'],
                "source_url": src['url'],
                "source_domain": src['domain'],
                "retrieved_at": src['retrieved_at'],
                "text": c,
                "score": score
            })
            
    # Sort chunks by relevance score (descending)
    all_chunks.sort(key=lambda x: x['score'], reverse=True)
    
    # Select top chunks while enforcing context size limits
    selected_chunks = []
    current_len = 0
    
    for chunk in all_chunks:
        chunk_len = len(chunk['text'])
        if current_len + chunk_len > max_chars:
            continue
            
        # We only want to include chunks that have *some* relevance, 
        # or at least the first chunk if nothing matches well.
        if chunk['score'] > 0 or len(selected_chunks) == 0:
            selected_chunks.append(chunk)
            current_len += chunk_len
            
    return selected_chunks
