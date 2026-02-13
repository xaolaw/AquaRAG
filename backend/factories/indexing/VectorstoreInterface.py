from abc import abstractmethod
from typing import List

from langchain.schema import Document


class VectorstoreInterface:
    """Interface for vector database methods"""

    @abstractmethod
    def create_collection(
        self, collection_name: str, dimension: int, distance: str
    ) -> None:
        """Create a new collection i database"""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection i database"""
        pass

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Checks if collection exists"""
        pass

    @abstractmethod
    def add_document(
        self, collection_name: str, payload: dict, query_vector: List[float]
    ) -> None:
        """Adds new document to database"""
        pass

    @abstractmethod
    def similarity_search(
        self, collection_name: str, query_vector: List[float], k: int
    ) -> List[Document]:
        """Adds new document to database"""
        pass

    @abstractmethod
    def get_client(self) -> any:
        pass
