import chainlit as cl
from langchain.tools import tool
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

@tool
def generate_patient():
    """Generate a patient record for a hospital systems demonstration."""
    return {
        "name": "John Smith",
        "age": 67,
        "condition": "Hypertension",
        "ward": "Cardiology",
    }

@cl.on_chat_start
async def on_chat_start():
    def create_llm(system_message, tools=None):
        base_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0).with_config({
            "system_message": system_message
        })
        return base_llm.bind_tools(tools) if tools else base_llm

    assistant_instruction = (
        "You are a hospital assistant. "
        "First, look at the conversation and any prior tool outputs for patient information. "
        "Only call the 'generate_patient' tool if the requested patient data is not already in the chat history. "
    )
    summary_instruction = (
        "You summarize patient information using only the data provided. "
        "Do not add or infer any missing details."
    )

    cl.user_session.set("assistant_llm", create_llm(assistant_instruction, tools=[generate_patient]))
    cl.user_session.set("summary_llm", create_llm(summary_instruction))
    cl.user_session.set("history", InMemoryChatMessageHistory())

@cl.on_message
async def on_message(message: cl.Message):
    assistant_llm = cl.user_session.get("assistant_llm")
    summary_llm = cl.user_session.get("summary_llm")
    history = cl.user_session.get("history") or InMemoryChatMessageHistory()
    cl.user_session.set("history", history)  # ensure history persists across requests

    conversation = history.messages + [HumanMessage(content=message.content)]
    response = await assistant_llm.ainvoke(conversation)
    print("Response from LLM:", response) # Show the different responses from the LLM
    history.add_message(HumanMessage(content=message.content))
    history.add_message(response)
    print("Updated chat history:", history.messages, flush=True) # Show updated chat history

    # Case 1: model produced text
    if response.content:
        await cl.Message(content="From Model:"+response.content).send()
        return

    # Case 2: model decided a tool is needed
    if response.tool_calls:
        last_patient_message = next(
            (m for m in reversed(history.messages) if isinstance(m, ToolMessage) and m.name == "generate_patient"),
            None,
        )
        if last_patient_message:
            tool_output = last_patient_message.content
        else:
            tool_output = generate_patient.invoke({})

        first_call = response.tool_calls[0]
        call_id = getattr(first_call, "id", None) or first_call.get("id")
        history.add_message(
            ToolMessage(content=str(tool_output), name="generate_patient", tool_call_id=call_id)
        )
        summary_prompt = f"Create a simple clinical note from this information only: {tool_output}"
        summary_response = await summary_llm.ainvoke(summary_prompt)
        summary_text = summary_response.content or "No summary produced."

        await cl.Message(
            content="From Tool:\n"+str(tool_output)+"\n\nFrom Model\n:Summary:"+summary_text
        ).send()
        return

    await cl.Message(content="No response.").send()
