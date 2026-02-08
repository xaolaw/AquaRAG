import re
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def read_pdf(path_to_pdf: str, mode: str = None) -> List[Document]:
    """
    Read a PDF (page by page) from the provided path.

    Args:
        path_to_pdf (str): Path to the PDF file.
        mode (str, optional): If defined, perform special segmentation.
            - article -> adds article number to every entry in metadata

    Returns:
        List[Document]: List of documents extracted from the PDF.

    Raises:
        ValueError: If the path is not a valid PDF.
    """
    try:
        loader = PyPDFLoader(file_path=path_to_pdf, mode="page")
        print("📄 Loading PDF file...")
        raw_docs = loader.load()
        documents = []
        previous_match = []

        for index, doc in enumerate(raw_docs):
            text = doc.page_content
            if mode == "article":
                match = re.findall(r"\bArt\.\s*(\d+)", text)

                article_number = match if match else [previous_match[-1]]

                if len(previous_match) > 0 and previous_match[-1] not in article_number:
                    article_number.insert(0, previous_match[-1])

                previous_match = article_number

                new_metadata = {
                    **doc.metadata,
                    "article_number": article_number,
                    "order": index,
                }
            else:
                new_metadata = doc.metadata

            documents.append(Document(page_content=text, metadata=new_metadata))

        return documents
    except FileNotFoundError as e:
        print(f"The provided file does not exist: {e}")
        raise

    except ValueError as e:
        print(f"The provided file is not a pdf: {e}")
        raise
