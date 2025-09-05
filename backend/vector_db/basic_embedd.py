import os
import re
from typing import List

from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()
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
        loader = PyPDFLoader(file_path=path_to_pdf, mode="page")
        print("📄 Loading PDF file...")
        raw_docs = loader.load()
        documents = []
        previous_match = []
        for index, doc in enumerate(raw_docs):
            text = doc.page_content
            # Match something like "Art. 123"
            match = re.findall(r"\bArt\.\s*(\d+)", text)

            # If there is no match we are still on previous article
            article_number = match if match else [previous_match[-1]]

            # Add the last article that we ended before
            if len(previous_match) > 0 and previous_match[-1] not in article_number:
                article_number.insert(0, previous_match[-1])

            previous_match = article_number

            new_metadata = {
                **doc.metadata,  # ← this keeps the original metadata! ✨
                "article_number": article_number,
                "order": index,
            }

            documents.append(Document(page_content=text, metadata=new_metadata))

        return documents
    except ValueError as e:
        print(
            "\033[91mValueError in ReadPdf: Provided path does not lead to a file: \033[0m",
            e,
        )
        return []


def EmbedDocument(file_path: str):
    """
    Using RecursiveCharacterTextSplitter we split loaded pdf by chunks where we pay attention and try to split them all by articles.
    After split we embed all of the chunks using NvidiaEmbedder
    """
    client = QdrantClient()

    if client.collection_exists(collection_name):
        print("🧹 Deleteing old child collection...")
        client.delete_collection(collection_name=collection_name)

    print("📦 Creating child collection...")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
        timeout=30,
    )
    doc_splitted = ReadPdf(file_path)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
        length_function=len,
    )

    print("📖 Splitting file...")
    split_docs = child_splitter.split_documents(doc_splitted)

    print(f"✅ Podzielono dokumenty na {len(split_docs)} chunków.")

    retriever = create_retriver()

    print("🧠 Adding documents to retriever...")
    retriever.add_documents(split_docs)

    return retriever


def create_retriver():
    """Creates parent retriver"""

    client = QdrantClient()

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=NVIDIAEmbeddings(model=os.environ["EMBEDDER"]),
    )

    print("🔍 Creating Retriever...")
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # albo "mmr"
        search_kwargs={"k": 3},  # liczba dokumentów
    )

    return retriever


load_dotenv()
if __name__ == "__main__":
    EmbedDocument("./data/prawod_wodne.pdf")
