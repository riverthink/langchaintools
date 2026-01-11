import chainlit as cl
from langchain.tools import tool
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
        "When asked about patient information, use the 'generate_patient' tool to retrieve patient records. "
    )
    summary_instruction = (
        "You summarize patient information using only the data provided. "
        "Do not add or infer any missing details."
    )

    cl.user_session.set("assistant_llm", create_llm(assistant_instruction, tools=[generate_patient]))
    cl.user_session.set("summary_llm", create_llm(summary_instruction))

@cl.on_message
async def on_message(message: cl.Message):
    assistant_llm = cl.user_session.get("assistant_llm")
    summary_llm = cl.user_session.get("summary_llm")

    response = await assistant_llm.ainvoke(message.content)
    print("Response from LLM:", response) # Show the different responses from the LLM

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
