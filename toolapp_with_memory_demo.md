# LangChain Memory with Chainlit Demo

**Files:** [toolapp_with_memory.py](./toolapp_with_memory.py)

## Overview
This demo showcases an interactive Chainlit application that demonstrates two critical LangChain concepts: **tool calling** and **conversation memory**. The application intelligently manages a mock hospital patient record system, avoiding redundant tool calls by checking conversation history.

## What This Demo Covers
- Creating and binding custom tools to LLMs
- Managing conversation history with `InMemoryChatMessageHistory`
- Building interactive Chainlit applications
- Implementing intelligent tool usage based on context
- Using multiple specialized LLMs for different tasks

## Architecture

The application uses a dual-LLM architecture:
1. **Assistant LLM**: Tool-enabled model that decides when to call `generate_patient`
2. **Summary LLM**: Focused model that formats patient data without tool access

## Code Structure

### 1. Tool Definition
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

The `@tool` decorator converts a regular Python function into a LangChain tool that can be:
- Discovered and called by the LLM
- Documented via its docstring
- Integrated into the conversation flow

**Important**: The docstring is critical—it tells the LLM what the tool does and when to use it.

### 2. Session Initialization
```python
@cl.on_chat_start
async def on_chat_start():
    def create_llm(system_message, tools=None):
        base_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0).with_config({
            "system_message": system_message
        })
        return base_llm.bind_tools(tools) if tools else base_llm
```

On chat start:
- Two LLMs are created with different system messages
- The assistant LLM is bound to the `generate_patient` tool
- An empty `InMemoryChatMessageHistory` is initialized
- All are stored in the session for persistence across messages

**Key Insight**: The assistant's system message instructs it to check chat history before calling the tool, preventing redundant API calls.

### 3. Conversation Memory Management
```python
@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get("history") or InMemoryChatMessageHistory()

    conversation = history.messages + [HumanMessage(content=message.content)]
    response = await assistant_llm.ainvoke(conversation)

    history.add_message(HumanMessage(content=message.content))
    history.add_message(response)
```

The conversation history pattern:
1. Retrieve existing history from session
2. Combine history with new user message
3. Send complete conversation to LLM
4. Store both user message and LLM response in history

This gives the LLM full context of the conversation, enabling it to:
- Reference previous information
- Avoid redundant tool calls
- Maintain conversational coherence

### 4. Tool Call Handling
```python
if response.tool_calls:
    last_patient_message = next(
        (m for m in reversed(history.messages) if isinstance(m, ToolMessage) and m.name == "generate_patient"),
        None,
    )
    if last_patient_message:
        tool_output = last_patient_message.content
    else:
        tool_output = generate_patient.invoke({})
```

Smart tool execution:
1. Check if the LLM wants to call a tool
2. Search conversation history for previous `generate_patient` results
3. Reuse existing data if found, otherwise call the tool
4. Add `ToolMessage` to history for future reference

**Why This Matters**: This pattern demonstrates how to build efficient agents that don't waste API calls or tool executions.

### 5. Response Handling
```python
if response.content:
    await cl.Message(content="From Model:"+response.content).send()
    return

if response.tool_calls:
    # ... tool handling ...
    summary_prompt = f"Create a simple clinical note from this information only: {tool_output}"
    summary_response = await summary_llm.ainvoke(summary_prompt)
```

The application handles two response types:
- **Direct text response**: LLM answered directly from conversation context
- **Tool call response**: LLM needs patient data, so we call the tool and then use the summary LLM to format it

## Conversation Flow Example

### First User Query
```
User: "Tell me about the patient"
→ Assistant LLM: Sees no patient data in history, calls generate_patient tool
→ Tool returns: {"name": "John Smith", "age": 67, ...}
→ Summary LLM: Formats the data into a clinical note
→ User sees: Patient data and formatted summary
```

### Follow-up Query
```
User: "What ward is the patient in?"
→ Assistant LLM: Sees patient data in conversation history
→ Directly responds: "The patient is in the Cardiology ward"
→ No tool call needed—data is already available
```

## Key LangChain Concepts Demonstrated

### 1. Tool Integration
```python
base_llm.bind_tools([generate_patient])
```
Binding tools makes them available to the LLM without hardcoding when they should be used—the LLM decides based on context.

### 2. Conversation Memory
```python
InMemoryChatMessageHistory()
```
Maintains message history across turns, enabling:
- Contextual understanding
- Follow-up questions
- Avoiding redundant operations

### 3. Multiple Message Types
- `HumanMessage`: User input
- `AIMessage`: LLM response (with or without tool calls)
- `ToolMessage`: Tool execution results

### 4. Dual-LLM Pattern
Using specialized LLMs for different tasks:
- Tool-enabled assistant for decision making
- Specialized summary LLM for formatting

## Running the Demo

```bash
chainlit run toolapp_with_memory.py --watch
```

Open http://localhost:8000 in your browser.

### Try These Interactions

1. **Initial query**: "Tell me about the patient"
   - Watch the tool being called

2. **Follow-up**: "What is their age?"
   - Notice no tool call—uses memory

3. **New session**: Refresh the page and ask again
   - Tool is called again (session was reset)

## Debug Output

The code includes print statements to show:
- LLM responses with tool calls
- Updated chat history after each turn

Check your terminal to see the internal state changes.

## Why This Pattern Matters

This demo shows production-ready patterns:
- **Efficiency**: Don't call tools unnecessarily
- **Context awareness**: Use conversation history intelligently
- **Separation of concerns**: Different LLMs for different tasks
- **State management**: Proper session handling in Chainlit

These patterns are essential for building real-world applications where:
- API calls cost money
- Users expect contextual conversations
- Tool executions may be expensive or rate-limited

## Next Steps

After understanding tools and memory:
1. Explore `ragdemo.py` to see how RAG extends this with document retrieval
2. Examine `travelplanner.py` to see complex multi-step orchestration
