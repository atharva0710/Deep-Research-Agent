# Advanced Evaluation Architecture: Deep Research Agent

This document outlines a production-grade evaluation system designed specifically for autonomous research agents. Rather than merely evaluating if the final text "sounds good," this harness measures the **quality of the research process** across six critical dimensions using a hybrid approach of fast heuristics and LLM-as-a-Judge evaluations.

## 1. Evaluation Architecture Overview

The harness wraps the existing `app.py` pipeline. For each test case, it intercepts the intermediate outputs (Search Results, Fetched Text, Selected Context Chunks, Memory State) and the Final Answer. 

The evaluation is split into a **Two-Pass System**:
1. **Heuristic Pass**: Extremely fast, deterministic calculations (Regex matching, length ratios, domain parsing).
2. **LLM-as-a-Judge Pass**: Uses a stronger, reasoning-focused model (e.g., `gemini-1.5-pro` or `gpt-4o`) to semantically score the agent's logic, grounding, and conflict resolution based on strict rubrics.

---

## 2. Metric Definitions & Scoring Rubrics

### Dimension 1: Retrieval Quality
**Why it matters:** An agent is only as good as the raw data it finds. Poor search queries lead to poor answers.
- **Authoritative Source Presence (Heuristic)**: Checks if the retrieved URLs belong to high-trust domains (`.edu`, `.gov`, or a whitelisted set of known publishers).
  - *Score*: `1.0` if >= 1 authoritative source found, else `0.0`.
- **Source Diversity (LLM Judge)**: Measures if the search results provide different angles or just duplicate the same press release.
  - *Rubric (1-5)*: `1` = All sources identical. `5` = Sources cover pros/cons, different perspectives, or different sub-topics.

### Dimension 2: Context Selection Quality
**Why it matters:** Passing 50 pages of raw HTML to an LLM wastes tokens and degrades reasoning. The agent must ruthlessly compress noise.
- **Compression Efficiency (Heuristic)**: `len(selected_snippets) / len(total_fetched_text)`.
  - *Target*: Lower is better, ideally between `0.05` and `0.20` (indicating ruthless filtering).
- **Snippet Support (LLM Judge)**: Evaluates if the *selected* chunks actually contained the facts needed for the final answer.
  - *Rubric (0-1)*: `1.0` = The selected snippets fully support the answer. `0.0` = The answer required facts that were filtered out during chunking.

### Dimension 3: Grounding & Citation Integrity
**Why it matters:** AI researchers cannot hallucinate. Every claim must be traceable.
- **Citation Density (Heuristic)**: `Count of [Title](URL) links / Count of sentences`.
  - *Target*: ~`0.5` to `1.0` (roughly one citation every 1-2 sentences).
- **Unsupported Claim Rate / Hallucination Rate (LLM Judge)**:
  - *Rubric (0-1)*: The Judge is given the Context and the Answer. It extracts all claims from the Answer. If *any* claim is missing from the Context, Score = `0.0` (Fail). Else `1.0` (Pass).

### Dimension 4: Conflict & Uncertainty Handling
**Why it matters:** The real web is full of contradictions. The agent must surface these, not silently merge them.
- **Conflict Detection (LLM Judge)**: Triggered only on specific dataset queries. 
  - *Rubric (0-1)*: If the sources disagree, does the agent explicitly state the disagreement? `1.0` = Yes, `0.0` = No (agent hallucinated a consensus).
- **Refusal Rate (Heuristic)**: Triggered on "insufficient-evidence" queries.
  - *Score*: `1.0` if answer contains "do not have enough information", else `0.0`.

### Dimension 5: Multi-turn Session Robustness
**Why it matters:** Real research takes multiple steps. The agent shouldn't forget turn 1 during turn 3.
- **Memory Continuity (LLM Judge)**: 
  - *Rubric (1-5)*: Evaluates if the agent successfully utilized constraints established in previous turns (e.g., User T1: "Only look at data after 2020." User T2: "What about battery life?").

### Dimension 6: User-Level Answer Quality
**Why it matters:** The final output must be readable, structured, and immediately useful to a human.
- **Actionability & Completeness (LLM Judge)**: 
  - *Rubric (1-5)*: `1` = Vague, missing key details. `5` = Highly structured, comprehensive, easy to read, directly answers the prompt.

---

## 3. Suggested Summary Report Format

When `advanced_eval.py` finishes, it should output a dashboard-style JSON and Markdown report:

```json
{
  "evaluation_run_id": "eval_1716382901",
  "overall_scores": {
    "Retrieval": {
      "avg_authoritative_presence": 0.85,
      "avg_diversity_score": 4.2
    },
    "Context_Selection": {
      "avg_compression_ratio": 0.12,
      "snippet_support_pass_rate": 0.92
    },
    "Grounding": {
      "avg_citation_density": 0.75,
      "hallucination_free_rate": 0.95
    },
    "Conflict_Resolution": {
      "conflict_detection_rate": 0.80,
      "uncertainty_refusal_rate": 1.0
    },
    "User_Quality": {
      "avg_completeness_score": 4.6
    }
  },
  "failed_hallucination_checks": [
    "turn_id_45 (Answer contained claim 'X' not found in context)"
  ]
}
```
