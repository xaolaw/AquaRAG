from typing import Dict

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()


class AquaRagEmbeddingFactory:
    def __init__(self, cfg: Dict):
        self.cfg = cfg["models"]["embedding"]

    def create(self) -> Embeddings:
        provider: str = self.cfg["provider"]
        name: str = self.cfg["name"]
        embedder = None

        if provider.lower() == "nvidia":
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

            try:
                embedder = NVIDIAEmbeddings(model=name)

            except Exception as e:
                raise ValueError(
                    f"There was a problem while creating Nvidia embedding model {name}: {e}"
                )

        else:
            raise ValueError(f"Unkown embedding provider: {provider}")

        return embedder
