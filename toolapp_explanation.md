# toolapp.py - Code Explanation

**Files:** [toolapp.py](./toolapp.py) | [Sequence Diagram](./sequencediagram.mermaid)

## Overview

This Python application demonstrates a **hospital assistant chatbot** built using Chainlit, LangChain, and OpenAI's GPT-4 model. The application showcases **tool calling** (also known as function calling), where the AI model can decide when to invoke specific tools to retrieve information rather than generating it from memory.

The app uses a **dual-LLM architecture**: one LLM acts as the assistant and decides when to call tools, while a second LLM takes the tool output and formats it into a human-readable summary.

## Dependencies

```python
import chainlit as cl
from langchain.tools import tool
from langchain_openai import ChatOpenAI
```

- **chainlit**: A framework for building conversational AI applications with a chat interface
- **langchain.tools**: Provides the `@tool` decorator to define tools that the LLM can call
- **langchain_openai**: LangChain's integration with OpenAI models

## Tool Definition

### `generate_patient()` - Lines 5-13

```python
@tool
def generate_patient():
    """Generate a patient record for a hospital systems demonstration."""
    return {
        "name": "John Smith",
        "age": 67,
        "condition": "Hypertension",
        "ward": "Cardiology",
    }
```

**Purpose**: This function is decorated with `@tool`, which transforms it into a tool that the LLM can invoke.

**How it works**:
- The `@tool` decorator converts this function into a structured tool description that the LLM can understand
- The docstring ("Generate a patient record...") is crucial - it tells the LLM **when** to use this tool
- The function returns a hardcoded patient dictionary with medical information
- In a real application, this would query a database instead of returning static data

**Tool Calling Flow**:
1. The LLM reads the docstring to understand what the tool does
2. When a user asks about patient information, the LLM decides to call this tool
3. The application executes the tool and captures the output
4. The output is passed to a second LLM for natural-language formatting

## Chat Initialization

### `on_chat_start()` - Lines 15-33

```python
@cl.on_chat_start
async def on_chat_start():
    def create_llm(system_message, tools=None):
        base_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0).with_config({
            "system_message": system_message
        })
        return base_llm.bind_tools(tools) if tools else base_llm

    assistant_instruction = (
        "You are a hospital assistant. "
        "When asked about patient information, use the 'generate_patient' tool to retrieve patient records. "
    )
    summary_instruction = (
        "You summarize patient information using only the data provided. "
        "Do not add or infer any missing details."
    )

    cl.user_session.set("assistant_llm", create_llm(assistant_instruction, tools=[generate_patient]))
    cl.user_session.set("summary_llm", create_llm(summary_instruction))
```

**Purpose**: This function runs once when a new chat session starts. It creates two specialized LLMs and stores them in the session.

**Step-by-step breakdown**:

1. **`create_llm()` helper (Lines 17-21)**: A local factory function that creates a configured LLM instance.
   - Always uses `gpt-4.1-mini` with temperature `0` (deterministic, no randomness)
   - Applies a system message to set the model's role and behaviour
   - Optionally binds tools using `bind_tools()` — only the assistant LLM receives tools

2. **`assistant_llm` (Line 32)**: The first LLM, configured with the `generate_patient` tool.
   - Its system message instructs it to use the tool when patient information is requested
   - `bind_tools([generate_patient])` gives it visibility of the tool schema so it can decide when to call it

3. **`summary_llm` (Line 33)**: The second LLM, with no tools attached.
   - Its system message focuses it on summarising provided data faithfully
   - It never calls tools — its only job is to format structured output into natural language

4. **Session storage**: Both LLMs are stored in `cl.user_session` so they are available to every subsequent message handler within this chat session.

## Message Handling

### `on_message()` - Lines 35-60

```python
@cl.on_message
async def on_message(message: cl.Message):
    assistant_llm = cl.user_session.get("assistant_llm")
    summary_llm = cl.user_session.get("summary_llm")

    response = await assistant_llm.ainvoke(message.content)
    print("Response from LLM:", response)

    # Case 1: model produced text
    if response.content:
        await cl.Message(content="From Model:"+response.content).send()
        return

    # Case 2: model decided a tool is needed
    if response.tool_calls:
        tool_output = generate_patient.invoke({})
        summary_prompt = f"Create a simple clinical note from this information only: {tool_output}"
        summary_response = await summary_llm.ainvoke(summary_prompt)
        summary_text = summary_response.content or "No summary produced."

        await cl.Message(
            content="From Tool:\n"+str(tool_output)+"\n\nFrom Model\n:Summary:"+summary_text
        ).send()
        return

    await cl.Message(content="No response.").send()
```

**Purpose**: This function handles every message the user sends.

**Step-by-step breakdown**:

1. **Lines 37-38**: Retrieves both LLMs from the session established at chat start.

2. **Line 40**: Invokes the `assistant_llm` with the user's message.
   - `ainvoke()` is the asynchronous form of invoke — it sends the message to the model and waits for the response without blocking the event loop.
   - The response will either contain `content` (a text reply) or `tool_calls` (a request to run a tool).

3. **Line 41**: Prints the raw response to the console — useful for seeing the different shapes a response can take during development.

4. **Lines 43-46 — Case 1: Text Response**
   - If `response.content` is non-empty, the model answered directly without needing a tool.
   - The text is sent to the user prefixed with "From Model:".
   - This occurs for general questions that don't require patient data.

5. **Lines 48-57 — Case 2: Tool Call followed by LLM Summary**
   - If `response.tool_calls` is set, the model has decided patient data is needed.
   - **Step 1 (Line 50)**: The application executes `generate_patient.invoke({})` to retrieve the patient record. This is the application's responsibility — the LLM does not execute tools itself, it only signals that a tool should be called.
   - **Step 2 (Lines 51-53)**: The raw tool output is passed to `summary_llm` with a prompt asking it to produce a clinical note. This is the key step: rather than showing raw JSON to the user, a second LLM interprets the structured data and writes it in natural language.
   - **Step 3 (Lines 55-57)**: Both the raw tool output and the formatted summary are sent to the user, making it clear where each piece of content came from.

6. **Line 60 — Fallback**: If neither `content` nor `tool_calls` are present, "No response." is sent. This is a safety catch for unexpected edge cases.

## Application Flow

### Sequence Diagram

```
User Message
     │
     ▼
assistant_llm.ainvoke()
     │
     ├── response.content set ──► Send text reply to user
     │
     └── response.tool_calls set
              │
              ▼
         generate_patient.invoke()   ← Application executes tool
              │
              ▼
         summary_llm.ainvoke(tool_output)  ← Second LLM formats output
              │
              ▼
         Send raw data + formatted summary to user
```

### Example Conversations

**Scenario 1: General question**
```
User: "Tell me about diabetes"
→ assistant_llm returns text content
→ Output: "From Model: Diabetes is a chronic condition where the body cannot properly regulate blood sugar levels. Type 1 diabetes occurs when the immune system attacks insulin-producing cells, while Type 2 is characterised by insulin resistance..."
```

**Scenario 2: Patient information request**
```
User: "Can you show me patient information?"
→ assistant_llm returns tool_calls
→ Application runs generate_patient.invoke({})
→ tool_output = {'name': 'John Smith', 'age': 67, 'condition': 'Hypertension', 'ward': 'Cardiology'}
→ summary_llm formats it into a clinical note
→ Output shows both raw data and the formatted summary
```

## Key Concepts Demonstrated

### 1. Tool Calling (Function Calling)
- The LLM decides autonomously when to use tools versus generating text
- Tools are defined with the `@tool` decorator; the docstring tells the LLM what the tool does
- The LLM does **not** execute tools — it signals intent, and the application executes

### 2. Dual-LLM Architecture
- **assistant_llm**: Has tools, handles routing and decision-making
- **summary_llm**: Has no tools, focused solely on formatting structured data into natural language
- Separating concerns keeps each LLM's system prompt focused and its behaviour predictable

### 3. Tool Output fed to a Summary LLM
- Raw tool output (a Python dictionary) is not shown to the user directly
- It is passed as input to `summary_llm`, which writes a natural-language clinical note from it
- This pattern — tool → structured output → LLM formatting — is common in production tool-calling systems

### 4. Session Management
- `cl.user_session.set()` and `get()` maintain state across messages
- Each chat session has its own pair of LLM instances

### 5. Asynchronous Processing
- `async`/`await` enables non-blocking operations
- Important for responsive chat applications handling concurrent users

### 6. Two-Stage Response Handling
- Stage 1: `assistant_llm` decides what to do (text reply or tool call)
- Stage 2: Application executes the decision, then optionally calls `summary_llm` to format the result

## Limitations and Potential Improvements

### Current Limitations

1. **Single Tool Execution**: Only handles one tool call per message; does not support chaining or multiple tool calls in a single turn
2. **Hardcoded Data**: Patient data is static, not from a real database
3. **No Error Handling**: No try/except blocks for API failures or tool execution errors
4. **No Conversation Memory**: Each message is handled independently; prior context is not retained

### Next Step: Adding Memory

`toolapp_with_memory.py` addresses the memory limitation by storing conversation history with `InMemoryChatMessageHistory`. It also avoids redundant tool calls by checking whether patient data was already retrieved earlier in the conversation.

## Use Cases

This pattern is useful for:

- **Medical Systems**: Querying patient records, lab results, appointments
- **Customer Service**: Looking up orders, account information, FAQs
- **Enterprise Apps**: Accessing databases, internal documentation, APIs
- **E-commerce**: Product searches, inventory checks, order status

## Conclusion

This application demonstrates the fundamental pattern of **LLM-driven tool calling**, where the model acts as an intelligent router that decides when to retrieve information via tools versus generating responses directly. The dual-LLM approach — one to decide, one to format — keeps each model's role clear and produces more natural output than forwarding raw tool data to the user. The `toolapp_with_memory.py` demo builds on this foundation by adding conversation memory.
