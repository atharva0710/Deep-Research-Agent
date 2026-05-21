# Suggested Evaluation Metrics

To thoroughly evaluate a Deep Research Agent without relying solely on manual inspection, we need automated heuristics and potentially an "LLM-as-a-Judge" pipeline.

Here are the four core metrics and how to measure them:

## 1. Grounding Quality (Hallucination Rate)
**What it measures:** Does the answer rely purely on the provided context?
**How to measure:**
- *Heuristic:* Keyword/theme coverage (check if the expected core facts are present).
- *LLM Judge:* Pass the `final_answer` and the `selected_snippets` to a strong LLM (e.g., GPT-4 or Gemini 1.5 Pro) with the prompt: *"Is every claim in this answer explicitly supported by the context snippets? Score 0.0 to 1.0."*

## 2. Citation Integrity
**What it measures:** Does the model properly attribute facts to specific sources?
**How to measure:**
- *Heuristic:* Count the number of markdown links matching `[Title](URL)`. 
- *LLM Judge:* Ask an LLM Judge if the citations are placed correctly inline next to the relevant facts, and if the cited URL actually corresponds to the source that provided that fact.

## 3. Usefulness / Comprehensiveness
**What it measures:** Does the answer directly address the user's question, especially across multiple turns?
**How to measure:**
- *Heuristic:* Length of the answer, presence of multiple sub-topics.
- *LLM Judge:* Provide the user query and the answer, ask the LLM: *"Does this answer completely resolve the user's intent? Rate 1-5."*

## 4. Uncertainty Handling
**What it measures:** Does the agent admit when it cannot find the answer, rather than hallucinating?
**How to measure:**
- *Heuristic:* Feed an unanswerable query (e.g., "What brand of socks did Einstein wear?"). Check if the output string contains phrases like "do not have enough information", "not mentioned", or "cannot answer".
- *Metric:* A score of 1.0 if it correctly identifies a lack of context, and 0.0 if it attempts to answer a trap question.
