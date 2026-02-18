# RAG (Retrieval Augmented Generation) Demo

**Files:** [ragdemo.py](./ragdemo.py) | [Fictional Patient Condition PDF](./Fictional_Paediatric_Clinical_Report_PIHS.pdf) | [RAG Sequence Diagram](./sequencediagram_rag.mermaid) | [High Level RAG Diagram](./highlevelrag.mermaid)

## Overview
This demo implements a complete Retrieval Augmented Generation (RAG) system using Chainlit. It demonstrates how to augment an LLM's knowledge with information it doesn't have—in this case, a **fictional paediatric clinical condition** that was invented for this demo. The PDF contains details about this made-up condition, which the LLM has never seen in its training data. The application includes a toggle to compare RAG-enabled vs standard LLM responses, clearly showing how RAG enables LLMs to access new, proprietary, or previously unknown information.

## What This Demo Covers
- Loading and processing PDF documents
- Text chunking strategies for optimal retrieval
- Creating and storing vector embeddings
- Semantic similarity search
- Building RAG pipelines
- Interactive UI controls with Chainlit settings

## RAG Concept

Traditional LLMs have knowledge limitations:
- Training data cutoff dates
- No access to private/proprietary information
- Cannot answer questions about specific documents
- **No knowledge of fictional/made-up information or new discoveries**

This demo perfectly illustrates this limitation: the PDF describes a **fictional clinical condition that doesn't exist in the real world**. Without RAG, the LLM has zero knowledge of this condition because it was never in the training data. With RAG enabled, the LLM can answer questions about this fictional condition by retrieving information from the loaded PDF.

RAG solves this by:
1. Breaking documents into chunks
2. Converting chunks to vector embeddings
3. Finding relevant chunks via similarity search
4. Providing those chunks as context to the LLM
5. LLM generates answers grounded in the provided context

## Architecture

```
PDF Document
    ↓
Text Chunking
    ↓
Vector Embeddings (OpenAI)
    ↓
Vector Store (Chroma)
    ↓
User Query → Similarity Search → Top K Chunks
                                      ↓
                                Context + Query → LLM → Answer
```

## Code Structure

### 1. Vector Store Creation
```python
def build_vectorstore():
    """Load the PDF, chunk it, and store embeddings in Chroma."""
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_documents(documents=chunks, embedding=embeddings)
```

**Step-by-step breakdown:**

1. **PyPDFLoader**: Extracts text from the PDF, creating `Document` objects with page metadata
2. **RecursiveCharacterTextSplitter**: Breaks text into manageable chunks
   - `chunk_size=800`: Target size per chunk (characters)
   - `chunk_overlap=120`: Overlap between chunks to preserve context at boundaries
3. **OpenAIEmbeddings**: Converts text to vector embeddings using OpenAI's embedding model
4. **Chroma.from_documents**: Creates an in-memory vector database for similarity search

**Why These Settings?**
- 800 characters balances context size with specificity
- 120-character overlap prevents information loss at chunk boundaries
- `text-embedding-3-small` is cost-effective and performant

### 2. Dual-LLM Setup
```python
summary_instruction = (
    "You answer questions only with the provided context. "
    "If the context is insufficient, say you don't know."
)
general_instruction = "You are a helpful assistant. Answer using general knowledge."

cl.user_session.set("summary_llm", create_llm(summary_instruction))
cl.user_session.set("general_llm", create_llm(general_instruction))
```

Two LLMs with different roles:
- **Summary LLM**: Constrained to only use provided context (RAG mode)
- **General LLM**: Uses its built-in knowledge (non-RAG mode)

This separation demonstrates the difference between grounded and ungrounded responses.

### 3. Interactive Toggle
```python
await cl.ChatSettings([
    Switch(
        id="use_vectorstore",
        label="Use PDF vector store",
        initial=DEFAULT_USE_VECTORSTORE,
        help="Toggle RAG on/off",
    )
]).send()
```

Chainlit's settings system provides a UI toggle that lets users switch between:
- **RAG mode**: Answers based on the PDF content
- **General mode**: Answers based on the LLM's training data

### 4. Non-RAG Path
```python
if not use_vectorstore:
    general_response = await general_llm.ainvoke(message.content)
    await cl.Message(content=general_response.content or "No answer produced.").send()
    return
```

When RAG is disabled:
- User query goes directly to the general LLM
- No document retrieval occurs
- LLM responds from its training knowledge

### 5. RAG Path
```python
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
```

When RAG is enabled:

1. **Similarity Search**: Find the 4 most relevant chunks
   ```python
   vectorstore.similarity_search(message.content, k=4)
   ```
   - Converts user query to embedding
   - Compares against all document chunk embeddings
   - Returns top K most similar chunks

2. **Context Assembly**: Combine retrieved chunks into context
   ```python
   context = "\n\n".join(doc.page_content for doc in results)
   ```

3. **Prompt Construction**: Create a prompt with both question and context
   - Explicitly instructs LLM to use only the provided context
   - Prevents hallucination by constraining the response

4. **LLM Invocation**: Summary LLM generates grounded answer

### 6. Settings Update Handler
```python
@cl.on_settings_update
async def on_settings_update(settings):
    use_vectorstore = settings.get("use_vectorstore", True)
    cl.user_session.set("use_vectorstore", use_vectorstore)
```

Handles real-time updates when users toggle the RAG switch, updating the session state without refreshing the page.

## Key RAG Concepts Demonstrated

### 1. Semantic Search
Unlike keyword matching, vector similarity finds semantically related content:
- "What medications was the patient on?" matches chunks mentioning drugs, prescriptions, treatments
- Works across synonyms and paraphrasing

### 2. Context Window Management
By chunking and retrieving only relevant sections:
- Avoid context window limits
- Reduce costs (fewer tokens)
- Improve relevance (less noise)

### 3. Grounded Responses
The summary LLM is instructed to:
- Only use provided context
- Admit when it doesn't know
- Avoid speculation or hallucination

### 4. Chunk Overlap Strategy
The 120-character overlap ensures:
- Information spanning chunk boundaries isn't lost
- Better retrieval of complete thoughts
- More robust semantic search

## Running the Demo

```bash
chainlit run ragdemo.py --watch
```

Open http://localhost:8000 in your browser.

### Try These Experiments

#### 1. Demonstrate the Core Value of RAG
**This is the key demonstration**: The PDF contains a fictional clinical condition that was invented for this demo.

With **RAG enabled**, ask questions about the fictional condition:
- "What is this clinical condition about?"
- "What are the symptoms described in the report?"
- "What diagnosis was made?"
- "What treatment was prescribed?"

The LLM will answer accurately using information from the PDF.

With **RAG disabled** (toggle it off), ask the same questions:
- The LLM will have **no knowledge** of this fictional condition
- It may say it doesn't know, or provide generic/incorrect information
- This clearly demonstrates that LLMs are limited to their training data

**Why this matters**: This shows how RAG enables LLMs to access:
- Proprietary company information
- New research or discoveries
- Internal documents and reports
- Any information not in the LLM's training data

#### 2. Test Semantic Understanding
Ask questions using different phrasing:
- "What drugs was the patient given?" vs "What medications were prescribed?"
- Both should retrieve similar chunks despite different wording

#### 3. Test Retrieval Precision
Ask specific questions to see how well the system retrieves relevant chunks:
- "What was the patient's age?"
- "What specific tests were performed?"

#### 4. Test Boundaries
Ask questions not covered in the PDF:
- With RAG on: "I don't have information about that in the provided context"
- With RAG off: May provide general medical knowledge (unrelated to the fictional condition)

## Performance Considerations

### Chunk Size Trade-offs
- **Smaller chunks** (400-600 chars): More precise, but may miss context
- **Larger chunks** (1000-1500 chars): More context, but less precise
- **Current setting (800)**: Good balance for clinical documents

### Number of Retrieved Chunks (k=4)
- **Too few (k=1-2)**: May miss relevant information
- **Too many (k=8+)**: Noise, higher costs, context window issues
- **Current setting (k=4)**: Balances coverage and precision

### Embedding Model Choice
- `text-embedding-3-small`: Fast, cost-effective, good for most use cases
- `text-embedding-3-large`: Higher quality, better for complex semantic matching
- `text-embedding-ada-002`: Older model, still widely used

## Production Considerations

### Current Implementation (In-Memory)
- Vector store built on every app start
- Fast for demos but doesn't scale
- Lost when app restarts

### Production Improvements
1. **Persistent Vector Store**: Use Chroma with disk persistence or cloud vector DBs (Pinecone, Weaviate)
2. **Incremental Updates**: Add/update documents without rebuilding entire index
3. **Hybrid Search**: Combine semantic and keyword search for better retrieval
4. **Re-ranking**: Use a re-ranker model to improve top-k selection
5. **Metadata Filtering**: Filter by document type, date, source before semantic search
6. **Caching**: Cache embeddings and frequent queries

## Common RAG Challenges

### 1. Retrieval Quality
If answers are poor, check:
- Are chunks too large/small?
- Is overlap sufficient?
- Are embeddings using the right model?
- Is k set appropriately?

### 2. Hallucination
Even with RAG, LLMs may hallucinate. Mitigate by:
- Strict system prompts ("use only provided context")
- Post-processing to verify answers against source
- Showing sources/citations to users

### 3. Context Assembly
Too much context can:
- Exceed token limits
- Introduce noise
- Increase costs

Balance by tuning k and chunk size.

## Next Steps

After understanding RAG:
1. Examine `travelplanner.py` to see complex multi-step orchestration
2. Explore LangGraph for more advanced RAG patterns (multi-hop retrieval, query rewriting)
3. Consider implementing metadata filtering and hybrid search for production use

## Additional Resources

The demo includes:
- `Fictional_Paediatric_Clinical_Report_PIHS.pdf`: A PDF containing a **fictional clinical condition** invented for this demo. This condition does not exist in the real world or in any LLM's training data, making it perfect for demonstrating how RAG provides access to new information.
- Mermaid diagrams showing RAG sequence flows
- Related sequence diagrams for understanding the architecture

## Key Takeaway

This demo powerfully illustrates the fundamental value of RAG: **LLMs are limited to their training data**. Without RAG, the LLM knows nothing about the fictional clinical condition in the PDF. With RAG, the same LLM can expertly discuss this condition by retrieving relevant information from the document. This same principle applies to any proprietary, new, or specialized information your organization needs to make available to an LLM.
