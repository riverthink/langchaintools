# Simple LangChain Introduction Demo

**Files:** [simple_langchain.py](./simple_langchain.py)

## Overview
This demo provides a foundational introduction to LangChain core concepts including prompts, chains, and structured outputs. It demonstrates how to build a simple yet powerful pipeline for processing clinical patient notes.

## What This Demo Covers
- Creating prompt templates with system and user messages
- Chaining prompts with language models using the pipe (`|`) operator
- Using structured outputs with Pydantic models
- Working with OpenAI's GPT models through LangChain

## Code Structure

### 1. Pydantic Model Definition
```python
class PatientNoteSummary(BaseModel):
    summary: str = Field(..., description="Clinical synopsis of the patient note")
    problems: list[str] = Field(..., description="Key problems or concerns mentioned")
```

The `PatientNoteSummary` model defines the expected output structure with two fields:
- `summary`: A concise clinical synopsis
- `problems`: A list of key problems or concerns

Using Pydantic ensures type safety and automatic validation of the LLM's output.

### 2. Prompt Template
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a clinician who writes concise, neutral patient note summaries."),
    ("user", "Summarize the patient note and list the key problems/concerns.\n\nNote:\n{note}"),
])
```

The `ChatPromptTemplate` creates a structured conversation with:
- A **system message** that sets the context and role (clinician)
- A **user message** with a placeholder `{note}` for input data

### 3. Chain Construction
```python
summarize_chain = prompt | ChatOpenAI(model="gpt-4.1-mini", temperature=0).with_structured_output(
    PatientNoteSummary
)
```

This is the core LangChain pattern using the pipe operator (`|`):
1. Start with the prompt template
2. Pipe it to the language model (`ChatOpenAI`)
3. Configure for structured output using the Pydantic model

**Key Configuration:**
- `model="gpt-4.1-mini"`: Uses OpenAI's efficient GPT-4 variant
- `temperature=0`: Ensures deterministic, consistent outputs
- `.with_structured_output(PatientNoteSummary)`: Forces the model to return data matching the Pydantic schema

### 4. Execution
```python
result = summarize_chain.invoke({"note": sample_note})
```

The chain is invoked with a dictionary containing the `note` key, which matches the placeholder in the prompt template.

## Sample Output

For the sample patient note:
```
76-year-old female with hypertension and CKD stage 3. Presented with
dizziness and BP 180/95. Started on amlodipine 5mg. Labs: creatinine
1.6 (baseline 1.5), potassium 4.8. No chest pain or neuro deficits.
```

The output will be a `PatientNoteSummary` object with:
```python
PatientNoteSummary(
    summary="76-year-old female with hypertension and CKD stage 3 presenting with elevated blood pressure (180/95) and dizziness.",
    problems=["Hypertension", "Chronic Kidney Disease Stage 3", "Dizziness", "Elevated Blood Pressure"]
)
```

## Key LangChain Concepts Demonstrated

### 1. Declarative Chain Building
The pipe operator creates readable, declarative chains:
```
Input → Prompt → LLM → Structured Output
```

### 2. Structured Outputs
Using `with_structured_output()` ensures the LLM returns properly formatted data that matches your Pydantic schema, making it easy to integrate with downstream systems.

### 3. Template Variables
The `{note}` placeholder in the prompt template is automatically filled when invoking the chain with matching keys in the input dictionary.

## Running the Demo

```bash
python simple_langchain.py
```

Make sure you have set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

## Why This Matters

This simple example demonstrates the power of LangChain's abstractions:
- No manual API calls
- Type-safe outputs
- Reusable components
- Easy to test and modify

These patterns scale to much more complex workflows, as you'll see in the other demos.

## Next Steps

After understanding this basic chain:
1. Explore `toolapp_with_memory.py` to see how tools and memory work
2. Learn about RAG in `ragdemo.py`
3. See complex orchestration in `travelplanner.py`
