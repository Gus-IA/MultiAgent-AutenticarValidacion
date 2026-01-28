import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


os.getenv("TAVILY_API_KEY")[:10] + "*" * 10

# herramienta tavilysearch
tool = TavilySearch(max_results=3)

# definimos el llm
llm = init_chat_model("openai:gpt-5.2")

# le asignamos las herramientas
llm_with_tools = llm.bind_tools([tool])


# creamos el grafo
class State(TypedDict):
    messages: Annotated[list, add_messages]


# creamos el grafo del agente
def agent(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


tool_node = ToolNode(tools=[tool])

# creamos el grafo
graph_builder = StateGraph(State)
graph_builder.add_edge(START, "agent")
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")
graph = graph_builder.compile()

# visualización del grafo
Image(graph.get_graph().draw_mermaid_png())

# bucle del agente con tools
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
