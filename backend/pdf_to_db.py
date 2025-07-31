import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

collection_name = os.environ["COLLECTION_NAME"]

"""
This file is responsible for loading a pdf file from hugginface, transforming it to embeded version and inserting as new collection to qdrant database
"""

# [m for m in NVIDIAEmbeddings.get_available_models() if "embed" in m.id]


def ReadPdf(path_to_pdf: str) -> List[Document]:
    """
    This function returns list of langchain documents.
    Loading whole file as one document in order to split it with chunk overlap.
    """
    try:
        loader = PyPDFLoader(file_path=path_to_pdf, mode="single")
        print("Loading pdf file...")
        document = loader.load()
        return document

    except ValueError as e:
        print(
            "\033[91mValueError in ReadPdf: Provided path does not lead to a file: \033[0m",
            e,
        )
        return []


def EmbedDocument(file_path: str) -> Tuple[List[List[float]], List[Document]]:
    """
    Using RecursiveCharacterTextSplitter we split loaded pdf by chunks where we pay attention and try to split them all by articles.
    After split we embed all of the chunks using NvidiaEmbedder
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=True,
        separators=["\n\n", "Art.[0-9]*", "\n"],
    )

    doc = ReadPdf(file_path)
    doc_splitted = splitter.split_documents(documents=doc)
    doc_splitted_txt = [doc.page_content for doc in doc_splitted]

    print("Embedding...")
    embedder = NVIDIAEmbeddings(model=os.environ["EMBEDDER"])
    embeddings = embedder.embed_documents(doc_splitted_txt)
    # print("First vector: ", embeddings[0])
    return embeddings, doc_splitted


def InsertToVectorDb(
    embeddings: List[List[float]], doc_splitted: List[Document]
) -> bool:
    client = QdrantClient(url=os.getenv("QDRANT_ADDRESS"))

    if client.collection_exists(collection_name):
        print("Deleteing old collection...")
        client.delete_collection(collection_name=collection_name)

    print("Creating collection...")
    max_len = -1
    for i in embeddings:
        max_len = max(max_len, len(i))

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=max_len, distance=Distance.COSINE),
        timeout=30,
    )

    print("Starting adding new vectors")
    for index, chunk in enumerate(doc_splitted):
        if index % 100 == 0:
            print(f"Runnig {round(index / len(doc_splitted), 4) * 100}%")
        try:
            point = PointStruct(
                id=index, vector=embeddings[index], payload={"content": chunk}
            )
        except Exception as e:
            print(index, e)

        client.upsert(collection_name=collection_name, points=[point])
    client.close()


load_dotenv()
embeddings, doc_splitted = EmbedDocument("./data/prawod_wodne.pdf")
InsertToVectorDb(embeddings, doc_splitted)
