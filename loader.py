from enum import Enum
from typing import List, Callable, Optional
from uuid import uuid4

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter as RCTS

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)

import settings


class DocType(Enum):
    DOCX = '.docx'
    PDF = '.pdf'
    TXT = '.txt'
    HTML = '.html'
    MD = '.md'


class DocumentLoader:
    LOADER_MAP: dict[DocType, Callable] = {
        DocType.PDF: PyPDFLoader,
        DocType.TXT: TextLoader,
        DocType.DOCX: Docx2txtLoader,
        DocType.HTML: UnstructuredHTMLLoader,
        DocType.MD: UnstructuredMarkdownLoader,
    }

    @staticmethod
    def get_file_ext(filepath: str) -> str:
        return filepath.split('.')[-1]

    @classmethod
    def load_document(cls, filepath: str) -> List[Document]:
        file_ext = f".{cls.get_file_ext(filepath)}"

        try:
            doctype = DocType(file_ext)
        except ValueError:
            raise ValueError(f"Unsupported file type: {file_ext}")

        settings.logger.info("[LOADING] Document loading in progress...")
        loader_class = cls.LOADER_MAP.get(doctype)

        if loader_class is None:
            raise ValueError(f"No loader found for document type {doctype}")

        return loader_class(filepath).load()


class DocumentChunker:
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.text_splitter = RCTS.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=['\n\n', '\n', ' ', '']
        )

    def get_chunks(self, data: List[Document]) -> List[Document]:
        settings.logger.info(f"[CHUNKING] Document chunking in progress...")
        docs = self.text_splitter.split_documents(data)
        settings.logger.info(
            f"[DONE] Document chunking carried out in total {len(docs)} docs...")
        return docs


class VectorStore:
    def __init__(self, collection_name: str = 'doc_collection', persist_directory: str = './ragdb'):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    @staticmethod
    def get_doc_ids(documents: List[Document]) -> List[str]:
        return [str(uuid4()) for _ in range(len(documents))]

    def save_documents(self, documents: List[Document]):
        uuids = self.get_doc_ids(documents)
        settings.logger.info("[LOADING] Vectorstore loading in progress ...")
        self.vector_store.add_documents(
            documents=documents,
            ids=uuids
        )
        settings.logger.info("[DONE] Vectorstore document loading done!")

    def update_document(self, docid: str, document: Document):
        return self.vector_store.update_document(
            document_id=docid,
            document=document
        )

    def delete_documents(self, docids: List[str]):
        return self.vector_store.delete(ids=docids)

    def similarity_search(self, query_text: str, k: int = 3):
        return self.vector_store.similarity_search_with_relevance_scores(query_text, k=k)


class DocumentProcessor:
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.vector_store = VectorStore()

    def process_document(self, filepath: str):
        data = self.loader.load_document(filepath)
        chunks = self.chunker.get_chunks(data)
        self.vector_store.save_documents(chunks)

    def update_document(self, docid: str, document: Document):
        return self.vector_store.update_document(docid, document)

    def delete_documents(self, docids: List[str]):
        return self.vector_store.delete_documents(docids)

    def search_similar_documents(self, query_text: str, k: int = 3):
        return self.vector_store.similarity_search(query_text, k)


def main(docpath: str):
    processor = DocumentProcessor()
    processor.process_document(docpath)


if __name__ == '__main__':
    # Example usage
    main('/home/ram/rag_app/media/hypothetical_college_data.pdf')

    # Example similarity search
    # processor = DocumentProcessor()
    # query_text = "How much money college received in academic year 2022-2023?"
    # results = processor.search_similar_documents(query_text, k=3)
    # print(results)
    # results = processor.search_similar_documents(query_text, k=3)
    # print(results)
