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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.tools import tool
import sys

load_dotenv()


# creamos una herramienta para que el agente pregunte por más información
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


# añadimos el human assistance como herramienta
tools = [
    TavilySearch(max_results=3),
    human_assistance,
]

# instancia llm
llm = init_chat_model("openai:gpt-5.2")

# asignamos herramientas
llm_with_tools = llm.bind_tools(tools)


# instancia grafo
class State(TypedDict):
    messages: Annotated[list, add_messages]


# instancia grafo agente
def agent(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# nodo para las herramientas
tool_node = ToolNode(tools=tools)

# creamos el grafo
graph_builder = StateGraph(State)
graph_builder.add_edge(START, "agent")
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")
graph = graph_builder.compile(checkpointer=MemorySaver())

# visualización del grafo
Image(graph.get_graph().draw_mermaid_png())

# identificador para guardar en memoria
config = {"configurable": {"thread_id": "1"}}

# bucle del agente con tools
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,  # para seguir un hilo de conversación
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
    # Comprueba si el agente necesita más información
    while True:
        snapshot = graph.get_state(config)
        if snapshot.next and snapshot.next[0] == "tools":
            # espera input
            human_response = input("Human: ")
            if human_response.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                sys.exit(0)
            # sigue con la ejecución
            human_command = Command(resume={"data": human_response})
            events = graph.stream(human_command, config, stream_mode="values")
            # muestra por pantalla los mensajes
            for event in events:
                if "messages" in event:
                    event["messages"][-1].pretty_print()
        else:
            break
