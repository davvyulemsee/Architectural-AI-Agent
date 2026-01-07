from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os
from typing import TypedDict, List, Union, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# tools

@tool
def add(a:int, b:int):
    """"Addition function"""
    return a + b

@tool
def subtract(a:int, b:int):
    """"Subtraction function """
    return a - b

@tool
def multiply(a:int, b:int):
    """Multiplication function"""
    return a * b

tools = [add, subtract, multiply]

llm = ChatGroq(model="openai/gpt-oss-20b").bind_tools(tools)
print("Model initialized!")


def process_node(state:AgentState)->AgentState:
    system_prompt = SystemMessage(
        content="You are a helpful AI message. Answer my questions accordingly. Be joyful and have an enthusiastic tone. Use emojis."
    )

    response = llm.invoke([system_prompt] + state["messages"])

    # for chunk in llm.stream([system_prompt] + state["messages"]):
    #     print(chunk.content, end="", flush=True)

    print(llm.stream([system_prompt] + state["messages"]))

    # print(response)

    # print(state["messages"])

    return {"messages":[response]} #this basically returns the updated state

def to_proceed(state:AgentState):
    # print(state)
    # print(state['messages'][-1])
    if not state['messages'][-1].tool_calls :
        return 'stop'
    else:
        return "loop"

graph = StateGraph(AgentState)

graph.add_node("main_node", process_node)

tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)

graph.add_conditional_edges(
    "main_node",
    to_proceed,
    {
        "stop":END,
        "loop": "tool_node"
    }

)

graph.add_edge("tool_node", "main_node")


graph.set_entry_point("main_node")

app = graph.compile()
print("app compiled!")

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# user_input = input("Ask a question: ")

# "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please."

conversation_history = {"messages": []}

# query  = {"messages": [("user", user_input)]}
# [s.pretty_print() for s in app.invoke(query)['messages']]

# print_stream(app.stream(query , stream_mode="values"))

# conversation_history['messages'].append(HumanMessage(content = user_input))

while True:

    user_input = input("Ask a question: ")

    if user_input=="exit":
        break

    query = {"messages": [{"role": "user", "content": user_input}]}
    conversation_history['messages'].append(HumanMessage(content = user_input))

    # [s.pretty_print() for s in app.stream(conversation_history)['messages']]
    print_stream(app.stream(conversation_history , stream_mode="values"))

    # app.invoke(conversation_history)['messages'][-1].content.pretty_print()

    # response = app.invoke(conversation_history)
    # latest = response['messages'][-1]
    #
    # if isinstance(latest, AIMessage):
    #     print(latest.content)

    # conversation_history = response
