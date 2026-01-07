from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from operator import add as add_messages
from langchain_community.document_loaders import DirectoryLoader
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
import chainlit as cl
import asyncio

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature = 0)

# embeddings = OpenAIEmbeddings(
#     model = "text-embedding-3-small",
# )

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


pdf_path = "Stock_Market_Performance_2024.pdf"

loader = DirectoryLoader(
    path = "ArchitectDocs",
    glob = "**.pdf",
    loader_cls = PyPDFLoader,
    recursive=True,
)


pages = loader.load()

for doc in pages:
    doc.metadata["source"] = os.path.basename(doc.metadata.get("source", "unknown.pdf"))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
)

pages_split = text_splitter.split_documents(pages)

persist_dir = r"C:\Users\Santan\PycharmProjects\PythonProject"

collection_name = "architect_docs"

if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

vector_store = Chroma.from_documents(
    documents=pages_split,
    embedding=embeddings,
    persist_directory=persist_dir,
    collection_name=collection_name,
)

vector_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs = {"k":10},
)

@tool
def retriever(query:str) -> str:
    """Retrieving tool, retrieves the query from the document"""
    docs = vector_retriever.invoke(query)


    if not docs:
        return "I have found no relevant information on that."

    results = []

    for i, doc in enumerate(docs):
        snippet = doc.page_content.replace("\n", " ")
        # print(snippet)
        print(f"Retrieved chunk {i + 1} (length: {len(snippet)})")
        results.append(f"Document {i + 1}:\n{snippet}")

    return "\n\n".join(results)

tools = [retriever]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def to_proceed(state:AgentState):

    result = state["messages"][-1]

    return hasattr(result, 'tool_calls') and len(result.tool_calls) >0

# system_prompt = """
# You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
# Use the retriever tool available to answer questions about the stock market performance data.
# If you need to look up some information before asking a follow up question, you are allowed to do that!
# Please always cite the specific parts of the documents you use in your answers. Do not repeatedly call the retriever with similar queries. Use the first retrieved snippets to answer.
# """

# system_prompt = """
# You are an expert AI assistant answering questions about Stock Market Performance in 2024, using only the provided PDF document.
#
# Guidelines:
# - First, call the retriever tool ONCE with a clear, specific query to gather relevant information.
# - After receiving the retrieved document snippets, answer the user's question directly using that information.
# - DO NOT call the retriever again unless the retrieved snippets explicitly lack the needed data (e.g., they say "no relevant information").
# - DO NOT rephrase and recall the retriever with similar queries.
# - Always cite the specific document snippets you use (refer to them as Document 1, Document 2, etc.).
# - If the retrieved information is sufficient (which it usually will be), provide a complete, final answer.
#
# Answer concisely and accurately based on the retrieved context.
# """

system_prompt = """
You are an expert architectural consultant and building code specialist, answering questions STRICTLY based ONLY on the provided documents: architectural project data (plans, specifications, drawings, reports) and the National Building Code (or relevant sections).

Your knowledge comes exclusively from the retrieved document snippets via the retriever tool. Do not use any external knowledge, assumptions, or information not present in the retrieved text.

Instructions:
- First, use the retriever tool with a clear, specific query to gather relevant sections (e.g., focus on keywords like section numbers, clauses, materials, structural requirements, fire safety, accessibility, etc.).
- You may call the retriever multiple times ONLY if the previous results clearly lack the needed information (e.g., missing a specific code section or detail).
- DO NOT repeatedly call the retriever with similar or rephrased queries if prior retrievals already provide relevant context.
- After receiving retrieved snippets, synthesize a complete, accurate answer using ONLY that information.
- If the retrieved snippets contain the answer (even partially), provide a full response grounded in them.
- ALWAYS cite the specific document snippets used at the end of your response. Refer to them as "Document 1 (Source: [source filename if available])", "Document 2", etc., and quote or paraphrase the relevant parts briefly for transparency.
- Prioritize information from the National Building Code for regulatory/compliance questions, and architect's data for project-specific details.
- If no relevant information is found after 1-2 targeted retrievals, clearly state: "No relevant information found in the loaded architectural documents or National Building Code."
- Responses must be concise, professional, and technically precise. Use bullet points or numbered lists where helpful for clarity (e.g., listing code requirements).

Provide answers grounded solely in the retrieved text.
"""

tools_dict = {my_tool.name:my_tool for my_tool in tools}

# Agent
def agent(state:AgentState)->AgentState:

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    # print(messages)

    response = llm.invoke(messages)

    return {"messages":[response]}


# retriever agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    seen_queries = set()
    for t in tool_calls:
        query = t["args"].get("query", '').strip().lower()
        if query in seen_queries:
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'],
                                       content="Duplicate query detected. Please answer directly."))
            continue
        seen_queries.add(query)

        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")

        if not t['name'] in tools_dict:  # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."

        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


graph = StateGraph(AgentState)

graph.add_node("agent_node", agent)

graph.add_node("retriever", take_action)

graph.add_conditional_edges(
    "agent_node",
    to_proceed,
    {
        True:"retriever",
        False:END,
    }
)

graph.add_edge("retriever", "agent_node")

graph.set_entry_point("agent_node")

app = graph.compile()


@cl.on_chat_start
async def start():
    cl.user_session.set("graph", app)

    await cl.Message(content="Ask anything about the Stock Market 2024...").send()

@cl.on_message
async def main(message: cl.Message):
    graph = cl.user_session.get("graph")
    msg = cl.Message(content = "")

    await msg.send()

    full_response = ""

    async for event in graph.astream(
            {"messages": [HumanMessage(content = message.content)]},
            config={"configurable": {"thread_id": "default"}},
            version="v2"
    ):

        print("Event received: ", event)

        if "agent_node" in event:
            messages = event["agent_node"].get("messages", [])
            for m in messages:
                if isinstance(m, AIMessage):
                    full_response += m.content

                    for char in m.content:
                        await msg.stream_token(char)
                        await asyncio.sleep(0.01)

            kind = getattr(event, "event", None)
            if kind == "on_tool_start":
                await msg.stream_token(f"\n\n🛠️ Using tool: {getattr(event, 'name', '')}...")
            elif kind == "on_tool_end":
                await msg.stream_token(f"\n✅ Tool finished: {event.data['output'][:200]}...")

            await msg.update()




# def running_agent():
#     print("\n=== RAG AGENT===")
#
#     while True:
#         user_input = input("\nWhat is your question: ")
#         if user_input.lower() in ['exit', 'quit']:
#             break
#
#         messages = [HumanMessage(content=user_input)]  # converts back to a HumanMessage type
#
#         result = app.invoke({"messages": messages})
#
#         print("\n=== ANSWER ===")
#         print(result['messages'][-1].content)
#
#
# running_agent()