# Deep Research Agent — Run Instructions

It looks like you are using a Python virtual environment (`.venv`). Follow these exact steps to install the required packages in your active environment and run the different parts of the agent.

## 1. Install Dependencies
First, ensure you are in the project directory and your virtual environment is active, then install the required packages:

```bash
# Navigate to the project directory
cd /Users/atharvamandhaniya/Desktop/Academics4-2/PS2/SarvamProject/deep-research-agent

# Install all dependencies into your active environment
python -m pip install -r requirements.txt
```

*(Note: Since we switched to Groq, running this will install the newly added `groq` package to fix the `ModuleNotFoundError` you just encountered).*

---

## 2. Run the Streamlit Web Application
To launch the interactive chat interface in your browser:

```bash
python -m streamlit run streamlit_app.py
```
This will open the web app at `http://localhost:8501`.

---

## 3. Run the Command-Line Interface (CLI)
If you prefer to use the agent in your terminal instead of the web UI:

```bash
python app.py
```
*(Type your question at the prompt, and type `quit` or `q` to exit).*

---

## 4. Run the Advanced Evaluation Harness
To test the agent against the curated dataset using the Heuristic metrics and the `llama-3.3-70b-versatile` LLM-as-a-judge:

```bash
python advanced_eval.py
```
This will process the cases in `advanced_eval_dataset.json`, respect the 30 RPM rate limits, and output a detailed report to `advanced_eval_results.json`.

---

## Troubleshooting
If you still see a `ModuleNotFoundError` (like `No module named 'groq'`), ensure that the Python executable you are using is linked to the environment where you installed the dependencies. You can explicitly run:

```bash
/Users/atharvamandhaniya/Desktop/Academics4-2/PS2/SarvamProject/.venv/bin/python -m pip install -r requirements.txt
```
