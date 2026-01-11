# LangChain Demos

A collection of four progressive LangChain demonstrations, from basic concepts to complex orchestration patterns.

## Demos Overview

### 1. Simple LangChain Introduction (`simple_langchain.py`)
A foundational example showing core LangChain concepts including prompts, chains, and structured outputs using Pydantic models. This demo summarizes clinical patient notes and extracts key problems.

[Detailed documentation](./simple_langchain_demo.md)

### 2. LangChain Memory with Chainlit (`toolapp_with_memory.py`)
An interactive Chainlit application demonstrating tool calling with conversation memory. Shows how to maintain chat history and avoid redundant tool calls by checking prior context.

[Detailed documentation](./toolapp_with_memory_demo.md)

### 3. RAG Demo (`ragdemo.py`)
A Retrieval Augmented Generation (RAG) implementation using Chainlit that demonstrates how RAG enables LLMs to access information they don't have. Uses a PDF containing a fictional clinical condition (invented for this demo) that doesn't exist in the LLM's training data. Includes a toggle to compare RAG vs non-RAG responses, clearly showing that without RAG, the LLM has no knowledge of the fictional condition.

[Detailed documentation](./ragdemo_demo.md)

### 4. Travel Planning Chain (`travelplanner.py`)
A multi-step chain demonstrating complex orchestration (brainstorm → outline → structured plan). This example intentionally shows how LangChain chains can become complex, serving as motivation for switching to LangGraph for advanced orchestration needs.

[Detailed documentation](./travelplanner_demo.md)

## Prerequisites
- Python 3.10+ recommended
- OpenAI API key with access to `gpt-4.1-mini`

## Installation
1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key (replace the value with your key):
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

## Running the Demos

### Simple LangChain Introduction
```bash
python simple_langchain.py
```

### LangChain Memory Demo (Chainlit)
```bash
chainlit run toolapp_with_memory.py --watch
```
Open http://localhost:8000 in your browser.

### RAG Demo (Chainlit)
```bash
chainlit run ragdemo.py --watch
```
Open http://localhost:8000 in your browser.

### Travel Planning Chain
```bash
python travelplanner.py
```

## Learning Path

These demos are designed to be explored in order:
1. Start with **simple_langchain.py** to understand basic chains and structured outputs
2. Progress to **toolapp_with_memory.py** to learn about tools and memory
3. Explore **ragdemo.py** to see how RAG works with vector embeddings
4. Finally, examine **travelplanner.py** to understand complex orchestration and why LangGraph becomes necessary

## Additional Resources
- Mermaid diagrams showing sequence flows and architecture
- Fictional clinical PDF demonstrating RAG with information unknown to the LLM
- Detailed markdown documentation for each demo
