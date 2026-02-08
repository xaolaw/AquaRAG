import os
from typing import Dict

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_core.retrievers import BaseRetriever
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from MyQdrantVectorstore import MyQdrantVectorstore


class AquaRetriever:
    def __init__(self, cfg: Dict):
        self.cfg_url = cfg["database"]["url"]
        self.cfg_type = cfg["database"]["type"]
        self.cfg_indexing = cfg["database"]["indexing"]
        self.cfg_recreate = cfg["database"]["indexing"]["recreate_collection"]
        if cfg["database"]["type"] == "qdrant":
            self.client = MyQdrantVectorstore(cfg["database"]["url"])

    def create_retriever(self) -> BaseRetriever:
        if self.cfg_type == "qdrant":
            if self.cfg_indexing["strategy"] == "parent":
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
                    client=self.client.get_client(),
                    collection_name=self.cfg_indexing["parent"]["child_collection"],
                    embedding=NVIDIAEmbeddings(model=os.environ["EMBEDDER"]),
                )

                fs = LocalFileStore(
                    f"D:\Studia\AquaRAG\{self.cfg_indexing['parent']['parent_collection']}"
                )
                docstore = create_kv_docstore(fs)

                print("🔍 Creating ParentDocumentRetriever...")
                retriever = ParentDocumentRetriever(
                    vectorstore=vectorstore,
                    docstore=docstore,
                    child_splitter=child_splitter,
                    parent_splitter=parent_splitter,
                )

        return retriever
