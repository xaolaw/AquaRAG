import logging
import os
import uuid
from typing import List

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

load_dotenv()
collection_name = os.environ["COLLECTION_NAME"]
collection_name_memories = os.environ["COLLECTION_MEMORIES"]
model = ChatOpenAI(model=os.environ["LLM"], temperature=0)


def vectorize_user_query(query: str) -> List[float]:
    """Embeds user_query vector"""
    embedder = NVIDIAEmbeddings(model=os.environ["EMBEDDER"])
    return embedder.embed_query(query)


def multi_query(user_query: str, number_of_queries: int) -> str:
    """Using user query creates a multiple versions of query for better document gather"""

    MULTI_QUERY_PROMPT = (
        "Jesteś asystentem AI. Twoim zadaniem jest wygenerowanie {number_of_queries} różnych wersji podanego pytania w celu późniejszego pobrania danych z bazy danych."
        "Poprzez wygenerowanie {number_of_queries} różnych wersji jego pytania, masz pomóc użytkownik na przezwyciężenie różnych trudności w wyszukaniu danych jakie mogą wynikać z jego pytania."
        "Podaj tylko i wyłącznie nowe pytania odseparowanie poprzez nowe linie, podaj to w taki sposób"
        "Oto {number_of_queries} nowe pytania"
        "Pytanie1: treść pytania"
        "Pytanie2: treść pytania"
        "itp."
        "Oryginalne pytanie {user_query}"
    )
    prompt = MULTI_QUERY_PROMPT.format(
        number_of_queries=number_of_queries, user_query=user_query
    )
    model_response = model.invoke([{"role": "assistant", "content": prompt}]).content
    return model_response.split("\n\n")[-number_of_queries:]


def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")

    return user_id


@tool
def retrive_data_from_db(user_query: str) -> str:
    """Using user query input returns data from vector database as a string of anserws from document"""

    multi_query_list = multi_query(user_query, 3)
    logging.info("retrive tool: ", multi_query_list)

    client = QdrantClient(url=os.getenv("QDRANT_ADDRESS"))
    response = []
    for index, query in enumerate(multi_query_list):
        clean_query = query.replace(f"Pytanie{index + 1}: ", "")

        points = client.query_points(
            collection_name=collection_name,
            query=vectorize_user_query(query=clean_query),
            limit=5,
            with_payload=True,
        ).points

        logging.info("retrive tool: ", points)
        response.extend(points)

    client.close()

    if len(response) == 0:
        return "Nie znalazłem dokumentów odpowiednich do zadanego pytania"

    response = sorted(response, key=lambda p: p.score, reverse=True)
    seen_id = set()
    filtered_list = []

    for point in response:
        if point.id not in seen_id:
            seen_id.add(point.id)
            filtered_list.append(point)

    response = filtered_list[:5]

    return "\n-----------\n".join(
        [i.payload["content"]["page_content"] for i in response]
    )


@tool
def recall_memory(memory: str, config: RunnableConfig) -> str:
    """Searches db for chat history for answering user request"""

    client = QdrantClient(url=os.getenv("QDRANT_ADDRESS"))
    user_id = get_user_id(config=config)

    memory_response = client.query_points(
        collection_name=collection_name_memories,
        query=vectorize_user_query(memory),
        limit=3,
        with_payload=True,
        query_filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        ),
    ).points

    if len(memory_response) == 0:
        return "Brak pamięci związanej z zapytaniem"

    client.close()

    return "\n".join([i.payload["memory"] for i in memory_response])
