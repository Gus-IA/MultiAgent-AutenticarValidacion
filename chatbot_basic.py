import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image

load_dotenv()


os.getenv("OPENAI_API_KEY")[:15] + "*" * 10

# inicializamos el llm
llm = init_chat_model("openai:gpt-5.2")


# estado del grafo
class State(TypedDict):
    messages: Annotated[list, add_messages]


# primer nodo del grafo
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# creamos el grafo
graph_builder = StateGraph(State)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# visualizamos el grafo
Image(graph.get_graph().draw_mermaid_png())

# bucle del chatbot
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
