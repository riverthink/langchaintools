# toolapp.py - Code Explanation

## Overview

This Python application demonstrates a **hospital assistant chatbot** built using Chainlit, LangChain, and OpenAI's GPT-4 model. The application showcases **tool calling** (also known as function calling), where the AI model can decide when to invoke specific tools to retrieve information rather than generating it from memory.

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

### `generate_patient()` - Line 5-13

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
3. The tool executes and returns the patient data
4. The application processes the returned data

## Chat Initialization

### `on_chat_start()` - Line 15-27

```python
@cl.on_chat_start
async def on_chat_start():
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    system_instruction = (
        "You are a hospital assistant. "
        "When asked about patient information, use the 'generate_patient' tool to retrieve patient records. "
    )
    llm_with_tools = llm.bind_tools([generate_patient]).with_config({
        "system_message": system_instruction
    })

    cl.user_session.set("llm", llm_with_tools)
```

**Purpose**: This function runs once when a new chat session starts.

**Step-by-step breakdown**:

1. **Line 17**: Creates an OpenAI LLM instance
   - Model: `gpt-4.1-mini` (a smaller, faster GPT-4 variant)
   - Temperature: `0` (deterministic responses, no randomness)

2. **Lines 19-22**: Defines system instructions
   - Sets the AI's role as a "hospital assistant"
   - Explicitly instructs it to use the `generate_patient` tool when needed
   - This guides the model's behavior throughout the conversation

3. **Lines 23-25**: Configures the LLM with tools
   - `bind_tools([generate_patient])`: Attaches the patient generation tool to the LLM
   - `with_config(...)`: Adds the system message configuration
   - This creates an LLM instance that knows about available tools

4. **Line 27**: Stores the configured LLM in the session
   - `cl.user_session.set()`: Stores data specific to this chat session
   - Allows the LLM to be retrieved in subsequent message handlers

## Message Handling

### `on_message()` - Line 29-47

```python
@cl.on_message
async def on_message(message: cl.Message):
    llm = cl.user_session.get("llm")

    response = await llm.ainvoke(message.content)
    print("Response from LLM:", response)

    # Case 1: model produced text
    if response.content:
        await cl.Message(content="From Model:"+response.content).send()
        return

    # Case 2: model decided a tool is needed
    if response.tool_calls:
        tool_output = generate_patient.invoke({})
        await cl.Message(content="From Tool:"+str(tool_output)).send()
        return

    await cl.Message(content="No response.").send()
```

**Purpose**: This function handles every message the user sends.

**Step-by-step breakdown**:

1. **Line 31**: Retrieves the LLM from the session
   - Gets the tool-enabled LLM configured in `on_chat_start()`

2. **Line 33**: Invokes the LLM with the user's message
   - `ainvoke()`: Asynchronous invocation of the LLM
   - Sends the user's message content to the model

3. **Line 34**: Debug output
   - Prints the raw response to the console
   - Helps developers see what the LLM returns (text vs. tool calls)

4. **Lines 37-39**: **Case 1 - Text Response**
   - Checks if the LLM returned text content
   - If yes, sends it to the user prefixed with "From Model:"
   - This happens when the LLM answers directly without using tools

5. **Lines 42-45**: **Case 2 - Tool Call**
   - Checks if the LLM requested to call a tool
   - If yes, executes the `generate_patient` tool
   - Sends the tool's output to the user prefixed with "From Tool:"
   - This demonstrates **two-stage processing**: LLM decides → Tool executes

6. **Line 47**: **Fallback**
   - If neither content nor tool_calls are present, sends "No response."
   - Safety catch for unexpected cases

## Application Flow

### Example Conversation Flow

**Scenario 1: User asks a general question**
```
User: "What services does the hospital provide?"
→ LLM responds with text
→ Output: "From Model: The hospital provides..."
```

**Scenario 2: User asks about patient information**
```
User: "Can you show me patient information?"
→ LLM decides to call generate_patient tool
→ Tool returns patient data
→ Output: "From Tool: {'name': 'John Smith', 'age': 67, ...}"
```

## Key Concepts Demonstrated

### 1. **Tool Calling (Function Calling)**
- The LLM can decide when to use tools vs. generating text
- Tools are defined using the `@tool` decorator
- The LLM receives tool descriptions and decides autonomously

### 2. **Session Management**
- `cl.user_session.set()` and `get()` maintain state across messages
- Each chat session has its own LLM instance

### 3. **Asynchronous Processing**
- `async`/`await` keywords enable non-blocking operations
- Important for responsive chat applications

### 4. **Two-Stage Response Handling**
- Stage 1: LLM decides what to do
- Stage 2: Application executes the decision (send text or call tool)

## Limitations and Potential Improvements

### Current Limitations

1. **No Tool Result Integration**: The tool output is sent directly to the user, but not fed back to the LLM for natural language formatting
2. **Single Tool Execution**: Only handles one tool call, doesn't support chains or multiple tool calls
3. **Hardcoded Data**: Patient data is static, not from a real database
4. **No Error Handling**: No try/catch blocks for API failures or tool execution errors

### Potential Improvements

```python
# Better approach: Feed tool result back to LLM
if response.tool_calls:
    tool_output = generate_patient.invoke({})
    # Create a message with tool results
    messages = [
        {"role": "user", "content": message.content},
        {"role": "assistant", "content": "", "tool_calls": response.tool_calls},
        {"role": "tool", "content": str(tool_output), "tool_call_id": response.tool_calls[0]["id"]}
    ]
    # Get LLM to format the tool result naturally
    final_response = await llm.ainvoke(messages)
    await cl.Message(content=final_response.content).send()
```

This would allow the LLM to present tool results in natural language rather than raw JSON.

## Use Cases

This pattern is useful for:

- **Medical Systems**: Querying patient records, lab results, appointments
- **Customer Service**: Looking up orders, account information, FAQs
- **Enterprise Apps**: Accessing databases, internal documentation, APIs
- **E-commerce**: Product searches, inventory checks, order status

## Conclusion

This application demonstrates the fundamental pattern of **LLM-driven tool calling**, where the model acts as an intelligent router that decides when to retrieve information via tools versus generating responses directly. While simplified for demonstration, this pattern scales to complex multi-tool applications with proper chaining and error handling.
