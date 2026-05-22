# Deep Research AI Agent: Assumptions, Links & Submission Report

This document outlines the core technical assumptions, model selections, architectural choices, and repository reference links for the **Deep Research Agent** challenge submission.

---

## 1. Project References & Links

* **GitHub Repository URL**: `[Insert your GitHub Repository Link here]`
* **Hosted Live Demonstration (Streamlit)**: `[Insert your Hosted Streamlit URL here, e.g., Streamlit Community Cloud]`
* **Inference Platform**: Groq Cloud API (High-performance serverless endpoints)
* **Search Engine Provider**: Tavily Search API

---

## 2. Core Architectural & Model Assumptions

### Inference Models
1. **Primary Synthesis & Reasoning Model**: `openai/gpt-oss-120b` (configured on Groq platform).
   * *Assumption*: High-parameter models are essential to follow strict grounding instructions, enforce structured markdown citation schemas, and execute nuanced conversational reasoning without hallucinating outside facts.
2. **LLM-as-a-Judge Evaluator**: `llama-3.3-70b-versatile`.
   * *Assumption*: Utilizing a separate high-capacity open-source model ensures objective grading of grounding, conflict detection, and conversational continuity while avoiding self-evaluation bias.

### Context Prioritization Criteria
To build a premium-grade research compiler rather than a basic keyword matching system, our context builder computes compound scores:
$$\text{Score} = (\text{TF Term Match} \times \text{Tavily Relevance Score}) \times \text{Recency Boost} \times \text{Source Diversity Discount}$$
* *Assumption*: Domain-diversity penalties (discounting a domain by $0.5^{\text{count}}$ for each chunk selected) prevent single-website echo chambers, forcing the agent to assemble answers from multiple distinct viewpoints.
* *Assumption*: Scanning and extracting current years (e.g., 2024–2026) within search snippets applies a positive multiplier ($1.2\times$), ensuring the model prioritizes real-time and up-to-date data.

### Session Persistence Layer
* *Assumption*: Using a local serverless **SQLite database** (`memory.db`) is superior to in-memory dictionary storage. It guarantees complete conversation log persistence across server restarts, maintains granular user/assistant transaction trace-logs, and allows modular storage of structured rolling summaries.

---

## 3. Context & Token Management Assumptions

* **Token Payload Budget**: The maximum payload allocated for model queries is capped at **12,000 characters** (roughly 3,000 tokens) to ensure fast inference speeds and stay within API rate limit windows.
* **Fallback Pruning Sequence**:
  If the combined text (Rolling Summary + Chat Logs + Web Context) exceeds 12,000 characters, the agent executes a structured fallback hierarchy:
  1. *Step 1*: Drop older conversation turns from the raw chat history.
  2. *Step 2*: If still exceeding, drop raw chat logs entirely and present only the SQLite **Rolling Session Summary**.
  3. *Step 3*: If still exceeding, progressively prune the lowest-ranked web context chunks one by one.

---

## 4. Evaluation Rubrics & Metrics Definition

The agent's performance was benchmarked using a hybrid evaluation suite implementing both heuristic and LLM-as-a-Judge methodologies:

### Heuristic Metrics
1. **Citation Density**: Ratio of correctly structured markdown inline citations (`[Title - Domain](URL)`) to total sentences. Capped at 1.0.
2. **Authoritative Presence**: Binary flag detecting the presence of trusted academic, government, or scientific domains (e.g., `.edu`, `.gov`, `nature.com`).
3. **Source Diversity**: Evaluated based on the unique domains cited in the final context. Scores: $\ge 3$ domains = 1.0; 2 domains = 0.7; 1 domain = 0.4; 0 domains = 0.0.

### LLM-as-a-Judge Rubrics
1. **Grounding Score**: The judge model evaluates the synthesized answer against raw context chunks. Any factual claim not strictly present in the source chunks results in a failure score (0.0), ensuring a strict zero-hallucination baseline.
2. **Conflict Resolution**: For queries with conflicting academic or scientific viewpoints, the judge verifies if the agent explicitly called out the contradiction (1.0) or chose a single bias (0.0).
3. **Memory Continuity**: For conversational follow-ups, the judge validates if the agent successfully resolved coreferences and answered utilizing context from previous session turns.
