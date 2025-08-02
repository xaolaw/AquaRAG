import logging
import os
from typing import List

import prompts as my_prompts
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers.cross_encoder import CrossEncoder

load_dotenv()
collection_name = os.environ["COLLECTION_NAME"]
collection_name_memories = os.environ["COLLECTION_MEMORIES"]
model = ChatOpenAI(model=os.environ["LLM"], temperature=0)
cross_encoder_model = CrossEncoder(os.environ["CROSS_ENCODER_MODEL"])


def vectorize_user_query(query: str) -> List[float]:
    """Embeds user_query vector"""
    embedder = NVIDIAEmbeddings(model=os.environ["EMBEDDER"])
    return embedder.embed_query(query)


def multi_query(user_query: str, number_of_queries: int) -> str:
    """Using user query creates multiple versions of the query for better document retrieval."""

    prompt = my_prompts.model_multi_query.invoke(
        {"user_query": user_query, "number_of_queries": number_of_queries}
    )

    model_response = model.invoke(prompt).content

    logging.info("multi_query model response")
    logging.info(model_response)
    return model_response.split("\n\n")[-number_of_queries:]


def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")

    return user_id


@tool
def retrive_data_from_db(user_query: str) -> str:
    """Using user query input returns data from vector database as a string of anserws from document"""

    logging.info("retrive_data_from_db")
    logging.info("performing multi query....")
    multi_query_list = multi_query(user_query, 3)

    client = QdrantClient(url=os.getenv("QDRANT_ADDRESS"))
    response = []
    for index, query in enumerate(multi_query_list):
        clean_query = query.replace(f"Pytanie{index + 1}: ", "")
        logging.info(clean_query)

        points = client.query_points(
            collection_name=collection_name,
            query=vectorize_user_query(query=clean_query),
            limit=5,
            with_payload=True,
        ).points

        logging.info("retrive tool bet answers: ", points)
        response.extend([(clean_query, point) for point in points])

    client.close()

    if len(response) == 0:
        return "Nie znalazłem dokumentów odpowiednich do zadanego pytania"
    response = sorted(response, key=lambda p: p[1].score, reverse=True)
    seen_id = set()
    filtered_list = []

    logging.info("response list:")
    logging.info(str(response))

    for question, point in response:
        if point.id not in seen_id:
            seen_id.add(point.id)
            filtered_list.append((question, point.payload["content"]["page_content"]))

    logging.info("filtered_list:")
    logging.info(str(filtered_list))

    results = sorted(
        {
            idx: r for idx, r in enumerate(cross_encoder_model.predict(filtered_list))
        }.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    logging.info("results from reranking:")
    logging.info(str(results))

    return "\n\n".join(filtered_list[index][1] for index, _ in results[:5])


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
