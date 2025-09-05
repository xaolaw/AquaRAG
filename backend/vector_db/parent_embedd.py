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
child_collection_name = os.environ["CHILD_COLLECTION_NAME"]
parent_collection_name = os.environ["PARENT_COLLECTION_NAME"]

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


def EmbedDocument(file_path: str) -> ParentDocumentRetriever:
    """
    Using RecursiveCharacterTextSplitter we split loaded pdf by chunks where we pay attention and try to split them all by articles.
    After split we embed all of the chunks using NvidiaEmbedder
    """
    client = QdrantClient()

    if client.collection_exists(child_collection_name):
        print("🧹 Deleteing old child collection...")
        client.delete_collection(collection_name=child_collection_name)

    print("📦 Creating child collection...")

    client.create_collection(
        collection_name=child_collection_name,
        vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
        timeout=30,
    )
    doc_splitted = ReadPdf(file_path)
    print("📖 Splitting file...")

    retriever = create_parent_retriever()

    print("🧠 Adding documents to retriever...")
    retriever.add_documents(doc_splitted)

    return retriever


def create_parent_retriever() -> ParentDocumentRetriever:
    """Creates parent retriver ParentDocumentRetriever"""

    client = QdrantClient()

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
        length_function=len,
    )

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600 * 5,
        length_function=len,
        is_separator_regex=True,
        separators=["Art.\s[0-9]*"],
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=child_collection_name,
        embedding=NVIDIAEmbeddings(model=os.environ["EMBEDDER"]),
    )

    fs = LocalFileStore("D:\Studia\AquaRAG\parent_docstore")
    docstore = create_kv_docstore(fs)

    print("🔍 Creating ParentDocumentRetriever...")
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    return retriever


load_dotenv()
if __name__ == "__main__":
    EmbedDocument("./data/prawod_wodne.pdf")
