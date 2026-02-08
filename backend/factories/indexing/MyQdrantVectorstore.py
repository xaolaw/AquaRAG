import uuid
from typing import List

from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from VectorstoreInterface import VectorstoreInterface


class MyQdrantVectorstore(VectorstoreInterface):
    def __init__(self, url: str):
        self.client = QdrantClient(url=url)

    def create_collection(
        self, collection_name: str, dimension: int, distance: str
    ) -> None:
        """
        Creates qdrant collection with given name for qdrant.

        Args:
            collection_name (str): collection name.
            client (QdrantClient): qdrant client

        Raises:
            ValueError: If the was problem while creating collection
        """
        try:
            if self.client.collection_exists(collection_name):
                print(f"🧹 Deleteing old {collection_name} collection...")
                self.client.delete_collection(collection_name=collection_name)

        except UnexpectedResponse:
            raise KeyError(f"Error while deleting collection {collection_name}")
        try:
            print(f"📦 Creating {collection_name} collection...")

            distnace_dict = {"cosine": Distance.COSINE}

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=distnace_dict[distance],
                ),
                timeout=30,
            )
        except UnexpectedResponse:
            raise KeyError(f"Error while creating collection {collection_name}")

    def delete_collection(self, collection_name: str) -> None:
        self.client.delete_collection(collection_name=collection_name)

    def collection_exists(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name)

    def add_document(
        self, collection_name: str, payload: dict, query_vector: List[float]
    ) -> None:
        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=query_vector,
                    payload=payload,
                )
            ],
        )

    def similarity_search(
        self, collection_name: str, query_vector: List[float], k: int
    ) -> List[Document]:
        pass

    def get_client(self) -> any:
        return self.client
