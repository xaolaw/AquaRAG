import json
import logging
import os
import re
from typing import Annotated, List

import prompts as my_prompts
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.storage import LocalFileStore
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_mistralai import ChatMistralAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from qdrant_client import QdrantClient, models
from qdrant_client.models import FieldCondition, Filter, MatchValue
from reranker import PolishCrossEncoder
from vector_db.basic_embedd import create_retriver
from vector_db.parent_embedd import create_parent_retriever

load_dotenv()
collection_name = os.environ["COLLECTION_NAME"]
collection_name_memories = os.environ["COLLECTION_MEMORIES"]
"""model = ChatOpenAI(model=os.environ["LLM"], temperature=0)
model = ChatMistralAI(
    model=os.environ["MISTRAL_LLM"], api_key=os.environ["MISTRAL_API"]
)
model = ChatNVIDIA(model=os.environ["LLAMA_LLM"], temperature=0)
"""
model = ChatMistralAI(
    model=os.environ["MISTRAL_LLM"], api_key=os.environ["MISTRAL_API"]
)
retriever = create_parent_retriever()
# retriever = create_retriver()
reranker = PolishCrossEncoder(os.environ["CROSS_ENCODER_MODEL"])


def vectorize_user_query(query: str) -> List[float]:
    """Embeds user_query vector"""
    embedder = NVIDIAEmbeddings(model=os.environ["EMBEDDER"])
    return embedder.embed_query(query)


def get_parent_elements(keys_list: List[str], fs: LocalFileStore) -> List[Document]:
    doc_list = []
    for key in keys_list:
        value = fs.mget([key])[0]
        decoded = value.decode("utf-8")
        data = json.loads(decoded)
        kwargs = data.get("kwargs")
        doc = Document(**kwargs)
        doc_list.append(doc)

    return doc_list


def multi_query(user_query: str, number_of_queries: int) -> str:
    """Using user query creates multiple versions of the query for better document retrieval."""

    prompt = my_prompts.model_multi_query.invoke(
        {"user_query": user_query, "number_of_queries": number_of_queries}
    )

    # Chat open ai model_response = model.invoke(prompt).content
    model_response = model.invoke(prompt.messages[0].content).content

    logging.info("multi_query model response")
    logging.info(model_response)
    model_response = model_response.replace("\n\n", "\n")
    return model_response.split("\n")[-number_of_queries:]


def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")

    return user_id


@tool
def retrive_data_from_db(
    user_query: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> dict:
    """Na podstawie zapytania danego użytkownika pobieramy odpowiednie fragmenty dokumentów z bazy danych"""

    logging.info("retrive_data_from_db")
    logging.info("performing multi query....")
    multi_query_list = multi_query(user_query, 5)
    # multi_query_list = [user_query]
    logging.info(multi_query_list)
    response = []
    for index, query in enumerate(multi_query_list[:5]):
        if "Pytanie" in query:
            clean_query = query.replace(f"Pytanie{index + 1}: ", "")
            logging.info(clean_query)

            points = retriever.vectorstore.similarity_search(clean_query)

            response.extend(
                [(clean_query, point) for point in points]
            )  # Tuple(str, langchain document)

    if len(response) == 0:
        return "Nie znalazłem dokumentów odpowiednich do zadanego pytania"

    cross_encoder_score = reranker.score(response)

    logging.info("results from reranking:")
    logging.info(str(cross_encoder_score))

    seen = set()
    parent_doc_ids = []
    child_ids = []

    for i, _ in cross_encoder_score:
        doc_id = response[i][1].metadata["doc_id"]
        if doc_id not in seen:
            seen.add(doc_id)
            parent_doc_ids.append(doc_id)

    parent_text = (
        "\n\n".join(
            doc.page_content
            for doc in get_parent_elements(parent_doc_ids[:3], retriever.docstore.store)
        ),
    )
    # This is for basic retriver parent_text = ("\n\n".join(doc.page_content for doc in child_text[:5]),)
    return Command(
        update={
            "parent_ids": parent_doc_ids,
            "child_ids": child_ids,
            "messages": [ToolMessage(content=parent_text, tool_call_id=tool_call_id)],
            "retrival_tool": "db",
        }
    )


@tool
def recall_memory(memory: str, config: RunnableConfig) -> str:
    """Searches db for chat history for answering user request"""

    user_id = get_user_id(config=config)
    client = QdrantClient()

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


@tool
def retrive_article(
    article_number: int, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Searches db for specific article when user gives article number"""

    response, _ = retriever.vectorstore.client.scroll(
        collection_name=os.environ["CHILD_COLLECTION_NAME"],
        scroll_filter=models.Filter(
            should=[
                models.FieldCondition(
                    key="metadata.article_number[]",
                    match=models.MatchValue(value=str(article_number)),
                ),
            ],
        ),
        with_payload=True,
    )

    parent_ids = []
    child_doc_ids = []
    for i in response:
        parent_element = (
            i.payload["metadata"]["doc_id"],
            i.payload["metadata"]["order"],
        )
        if parent_element not in parent_ids:
            parent_ids.append(parent_element)
            # hild_doc_ids.append(i.payload["metadata"]["_id"])

    child_ids = []
    parent_ids = [x[0] for x in sorted(parent_ids, key=lambda x: x[1], reverse=False)]

    article_str = "\n\n".join(
        doc.page_content
        for doc in get_parent_elements(parent_ids, retriever.docstore.store)
    )

    matches = re.finditer(r"Art.\s*[0-9]*", article_str)

    start = -1
    end = -1

    for m in matches:
        if start == -1 and m.group(0) == f"Art. {str(article_number)}":
            start = m.start()
        elif m.group(0) == f"Art. {str(article_number + 1)}":
            end = m.start()
            break

    return Command(
        update={
            "parent_ids": parent_ids,
            "child_ids": child_ids,
            "messages": [
                ToolMessage(content=article_str[start:end], tool_call_id=tool_call_id)
            ],
            "retrival_tool": "article",
        }
    )
