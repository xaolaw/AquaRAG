import logging

import aqua_rag as aqua
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

logging.basicConfig(level=logging.INFO)


def toggle_logging(enable: bool):
    if enable:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.CRITICAL + 1)


toggle_logging(True)

graph = StateGraph(aqua.RagState)

graph.add_node(aqua.generate_query_context)
graph.add_node("retrieve", ToolNode(aqua.retrive_tools))
graph.add_node(aqua.generate_user_answer)
graph.add_node(aqua.save_memory)

graph.add_edge(START, "generate_query_context")
graph.add_conditional_edges(
    "generate_query_context",
    aqua.route_tools,
    {
        "tools": "retrieve",
        END: "save_memory",
    },
)
graph.add_edge("retrieve", "generate_user_answer")
graph.add_edge("generate_user_answer", "save_memory")
graph.add_edge("save_memory", END)

agent = graph.compile()

png_bytes = agent.get_graph().draw_mermaid_png()

with open("rag.png", "wb") as f:
    f.write(png_bytes)

config = {"configurable": {"user_id": "1", "thread_id": "1"}}

while True:
    user_input = input("Ty (senpai): ")

    if user_input.lower() in ["exit", "quit"]:
        print("Pa-Pa~! Nie zapomnij wrócić... b-baka 😳💔")
        break

    input_data = {"messages": user_input}

    for chunk in agent.stream(input_data, config=config):
        for node, update in chunk.items():
            print("Update from node", node)
            if len(update["messages"]) > 0:
                update["messages"][-1].pretty_print()
            print("\n\n")
