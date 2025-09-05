import json
import logging
import os
import uuid
from typing import Annotated, List, Optional, Sequence, TypedDict

import prompts as my_prompts
import tools as my_tools
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.graph.message import add_messages
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

load_dotenv()

collection_name_memories = os.environ["COLLECTION_MEMORIES"]


# TODO: to add Current context for user one user chat, add step for graph to load context of the chat
class RagState(TypedDict):
    """
    Basic State of Rag application cosists of array of recent messages and stored memory
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    short_conversation_context: str
    parent_ids: Optional[List[str]]
    child_ids: Optional[List[str]]
    user_query: Optional[str]
    retrival_tool: Optional[str]
    mode: Optional[str]


retrive_tools = [
    my_tools.recall_memory,
    my_tools.retrive_data_from_db,
    my_tools.retrive_article,
]
model = ChatOpenAI(model=os.environ["LLM"], temperature=0).bind_tools(retrive_tools)
model_chain = my_prompts.model_init_prompt | model


def generate_query_context(state: RagState) -> RagState:
    """
    Based on questions decides if it should use a retrival tool or answer normally"""
    """
    TODO: better guardrail
    system_message = SystemMessage(
        "Zdecyduj czy pytanie odnosi się do inżynieri wodnej, jeśli nie, powiedz: udzielam odpowiedzi tylko dotyczących prawa wodnego"
    )"""
    response = model_chain.invoke({"messages": state["messages"]})
    return {
        "messages": state["messages"] + [response],
        "user_query": state["messages"][0].content,
        "mode": state["mode"],
    }


def generate_user_answer(state: RagState) -> RagState:
    """
    Generates answer to user after gathering all information about question
    """

    GENERATE_PROMPT = (
        "Jesteś botem odpowiadającym na zapytania użytkownika na podstawie dokumentu prawa wodnego. Udało Ci się uzyskać następujące dokumenty z bazy danych: \n"
        " Document z bazy:\n{context}\n\n "
        " Z ich pomocą odpowiedz na następujące pytanie {question}\n"
        " Pamiętaj że masz teraz odpowiedzieć na pytanie nie używaj żadnych narzędzi (tools), ponieważ wcześniej już ich użyłeś\n"
    )
    msg = state["messages"][-1]
    if isinstance(msg, ToolMessage) and state["messages"][-1].name == "retrive_article":
        GENERATE_PROMPT = (
            "Jesteś botem odpowiadającym na zapytania użytkownika na podstawie dokumentu prawa wodnego. "
            " Użytkownik oczekuje że przytoczysz mu informacje o danym artykule, pamiętaj o tym aby przy przytaczaniu artykułu przytoczyć wszystkie pobrane informacje."
            " Artykuły mogą mieć informacje w taki sposób Art. 192 (tekst akrtykułu) Art. 192a (dalsza część artykułu) Art. 192.1 (dalsza część artykułu) masz uwzględnić wszystkie takie części"
            " Udało Ci się uzyskać następujące dokumenty z bazy danych na temat artykułu: \n"
            " Document z bazy:\n{context}\n\n "
            " Z ich pomocą odpowiedz na następujące pytanie {question}\n"
            " Pamiętaj że masz teraz odpowiedzieć na pytanie nie używaj żadnych narzędzi (tools), ponieważ wcześniej już ich użyłeś\n"
        )
    prompt = GENERATE_PROMPT.format(
        question=state["messages"][0], context=state["messages"][-1]
    )
    # logging.info(prompt)
    response = model.invoke(prompt)
    logging.info(msg)
    state["messages"] = [response]
    # state["parent_ids"] = msg.content["parent_ids"]
    # state["child_ids"] = msg.content["child_ids"]
    return state


def save_memory(state: RagState, config: RunnableConfig) -> RagState:
    """
    Saves chat messages to vector database. Before giving answer to the user save his and your response to database!
    """
    memory_to_save = "\n".join(
        item.content
        for item in state["messages"]
        if isinstance(item, (AIMessage, HumanMessage)) and item.content
    )
    print(state)
    json_to_save = {
        "question": state["user_query"],
        "system_answer": state["messages"][-1].content,
        "parent_ids": state["parent_ids"],
        "child_ids": state["child_ids"],
        "retrival_tool": state["retrival_tool"],
        "mode": state["mode"],
    }
    with open("./output/tmp.json", mode="a", encoding="UTF-8") as f:
        f.write(json.dumps(json_to_save, ensure_ascii=False) + "\n")
    f.close()
    client = QdrantClient()
    user_id = my_tools.get_user_id(config=config)

    memory_vector = my_tools.vectorize_user_query(memory_to_save)

    if not client.collection_exists(collection_name_memories):
        client.create_collection(
            collection_name=collection_name_memories,
            vectors_config=VectorParams(
                size=len(memory_vector), distance=Distance.COSINE
            ),
            timeout=30,
        )

    client.upsert(
        collection_name=collection_name_memories,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=memory_vector,
                payload={"user_id": user_id, "memory": memory_to_save},
            )
        ],
    )

    client.close()
    return {"messages": []}


def route_tools(state: RagState) -> RagState:
    """
    Determine whether to use tools or end the conversation based on the last message.
    """
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"

    return END
