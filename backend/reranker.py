from typing import List, Tuple

from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
from langchain.schema import Document
from sentence_transformers.cross_encoder import CrossEncoder


class PolishCrossEncoder(BaseCrossEncoder):
    def __init__(self, model_name_or_instance):
        if isinstance(model_name_or_instance, str):
            self.model = CrossEncoder(model_name_or_instance)
        else:
            self.model = model_name_or_instance

    def score(self, docs: List[Tuple[str, Document]]) -> List[Tuple[int, float]]:
        cross_encdoer_list = [(query, doc.page_content) for query, doc in docs]

        results = sorted(
            {
                idx: r for idx, r in enumerate(self.model.predict(cross_encdoer_list))
            }.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return results
