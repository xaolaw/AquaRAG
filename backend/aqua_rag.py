import logging
import os
from typing import Annotated, List, Sequence, TypedDict

import tools as tools_
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
logging.basicConfig(level=logging.INFO)


class RagState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


model = ChatOpenAI(model=os.environ["LLM"], model_kwargs={"temperature": 0})
tools = [tools_.retrive_data_from_db]


def generate_query_context(state: RagState) -> RagState:
    """Based on questions decides if it should use a retrival tool or answer normally"""
    """system_message = SystemMessage(
        "Zdecyduj czy pytanie odnosi się do inżynieri wodnej, jeśli nie, powiedz: udzielam odpowiedzi tylko dotyczących prawa wodnego"
    )"""
    response = model.bind_tools(tools).invoke(state["messages"])
    return {"messages": state["messages"] + [response]}


def generate_user_answer(state: RagState) -> RagState:
    """Generates answer to user after gathering all information about question"""

    GENERATE_PROMPT = (
        "Jesteś botem odpowiadającym na zapytania użytkownika na podstawie dokumentu prawa wodnego. Udało Ci się uzyskać następujące dokumenty z bazy danych: "
        "Document z bazy:\n{context}\n\n "
        "Z ich pomocą odpowiedz na następujące pytanie {question}\n"
    )
    prompt = GENERATE_PROMPT.format(
        question=state["messages"][0], context=state["messages"][-1]
    )
    logging.info(prompt)
    response = model.invoke([{"role": "assistant", "content": prompt}])
    return {"messages": [response]}


graph = StateGraph(RagState)

graph.add_node(generate_query_context)
graph.add_node("retrieve", ToolNode(tools))
graph.add_node(generate_user_answer)

graph.add_edge(START, "generate_query_context")
graph.add_conditional_edges(
    "generate_query_context",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)
graph.add_edge("retrieve", "generate_user_answer")
graph.add_edge("generate_user_answer", END)

agent = graph.compile()

input = {
    "messages": "Co się stanie jeśli właściciel urządzenia wodnego nie wystąpił z wnioskiem, októrym mowa  wust. 1?"
}
for chunk in agent.stream(input):
    for node, update in chunk.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")
