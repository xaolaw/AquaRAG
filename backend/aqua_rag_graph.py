from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import aqua_rag as aqua

graph = StateGraph(aqua.RagState)

graph.add_node(aqua.generate_query_context)
graph.add_node("retrieve", ToolNode(aqua.tools))
graph.add_node(aqua.generate_user_answer)

graph.add_edge(START, "generate_query_context")
graph.add_conditional_edges(
    "generate_query_context",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)
graph.add_edge("retrieve", "generate_user_answer")
graph.add_edge("generate_user_answer", END)

agent = graph.compile()

input = {
    "messages": "Co się stanie jeśli właściciel urządzenia wodnego nie wystąpił z wnioskiem, o którym mowa w ust. 1?"
}
for chunk in agent.stream(input):
    for node, update in chunk.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")
