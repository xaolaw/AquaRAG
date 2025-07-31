import logging
import os
from typing import List

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

load_dotenv()
collection_name = os.environ["COLLECTION_NAME"]
model = ChatOpenAI(model=os.environ["LLM"], model_kwargs={"temperature": 0})


def vectorize_user_query(query: str) -> List[float]:
    """Embeds user_query vector"""
    embedder = NVIDIAEmbeddings(model=os.environ["EMBEDDER"])
    return embedder.embed_query(query)


@tool
def retrive_data_from_db(user_query: str) -> str:
    """Using user query input returns data from vector database as a string of anserws from document"""

    multi_query_list = multi_query(user_query, 3)
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

        response.extend(points)

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

    return "\n".join([i.payload["content"]["page_content"] for i in response])


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
