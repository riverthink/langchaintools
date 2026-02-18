import chainlit as cl
from chainlit.input_widget import Switch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PDF_PATH = "Fictional_Paediatric_Clinical_Report_PIHS.pdf"
DEFAULT_USE_VECTORSTORE = True


def build_vectorstore():
    """Load the PDF, chunk it, and store embeddings in Chroma."""
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_documents(documents=chunks, embedding=embeddings)


def create_llm(system_message: str):
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    ).with_config({"system_message": system_message})


@cl.on_chat_start
async def on_chat_start():
    vectorstore = build_vectorstore()
    cl.user_session.set("vectorstore", vectorstore)
    cl.user_session.set("use_vectorstore", DEFAULT_USE_VECTORSTORE)

    summary_instruction = (
        "You answer questions only with the provided context. "
        "If the context is insufficient, say you don't know."
    )
    general_instruction = "You are a helpful assistant. Answer using general knowledge."
    cl.user_session.set("summary_llm", create_llm(summary_instruction))
    cl.user_session.set("general_llm", create_llm(general_instruction))

    await cl.ChatSettings(
        [
            Switch(
                id="use_vectorstore",
                label="Use PDF vector store",
                initial=DEFAULT_USE_VECTORSTORE,
                help="Toggle RAG on/off",
            )
        ]
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    vectorstore: Chroma = cl.user_session.get("vectorstore")
    summary_llm: ChatOpenAI = cl.user_session.get("summary_llm")
    general_llm: ChatOpenAI = cl.user_session.get("general_llm")
    use_vectorstore = cl.user_session.get("use_vectorstore", True)

    if vectorstore is None or summary_llm is None or general_llm is None:
        await cl.Message(content="App not initialized yet. Please restart.").send()
        return

    if not use_vectorstore:
        general_response = await general_llm.ainvoke(message.content)
        await cl.Message(content=general_response.content or "No answer produced.").send()
        return

    results = vectorstore.similarity_search(message.content, k=4)
    if not results:
        await cl.Message(
            content="No relevant information found in the document while RAG is enabled."
        ).send()
        return

    context = "\n\n".join(doc.page_content for doc in results)
    prompt = (
        "Use only the context to answer the user's question.\n\n"
        f"Question: {message.content}\n\n"
        f"Context:\n{context}"
    )

    summary = await summary_llm.ainvoke(prompt)
    answer_text = summary.content or "No answer produced."

    await cl.Message(content=answer_text).send()


@cl.on_settings_update
async def on_settings_update(settings):
    # Chainlit sends back the settings values as a dict keyed by id.
    use_vectorstore = settings.get("use_vectorstore", True)
    cl.user_session.set("use_vectorstore", use_vectorstore)
