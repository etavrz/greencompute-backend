import datetime
import os
import pathlib

import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document as LCDocument
from loguru import logger
from pydantic import BaseModel
from sqlalchemy.orm import Session
from textractor import Textractor
from textractor.data.constants import TextractFeatures
from textractor.data.text_linearization_config import TextLinearizationConfig
from transformers import AutoTokenizer

from greencompute_backend.db.engine import engine
from greencompute_backend.db.tables import Base, Document
from greencompute_backend.services.llm.svc import EMBEDDINGS_MODEL


class DocumentMetadata(BaseModel):
    title: str
    url: str | None = None


def text_extraction(document: pathlib.Path) -> str:
    """Extract text from a document using Amazon textract.

    Args:
        document (pathlib.Path): Path to the document to extract text from.

    Returns:
        str: Extracted text from the document
    """
    if os.path.exists(f"extraction/{document.stem}.txt"):
        logger.warning(f"File extraction/{document.stem}.txt already exists. Skipping.")
        with open(f"extraction/{document.stem}.txt", "r") as file:
            return file.read()
    else:
        extractor = Textractor(profile_name="default")
        config = TextLinearizationConfig(hide_figure_layout=True, title_prefix="# ", section_header_prefix="## ")
        logger.info(f"Beggining text extraction for {document}")
        extracted_content = extractor.start_document_analysis(
            file_source=document.as_posix(),
            s3_upload_path="s3://greencompute",
            features=[TextractFeatures.LAYOUT],
            save_image=True,
        )
        text = extracted_content.get_text(config=config)

        os.makedirs("extraction", exist_ok=True)
        with open(f"extraction/{document.stem}.txt", "w") as file:
            file.write(text)
            logger.debug(f"Text written to extraction/extract_{document.stem}.txt")

        return extracted_content.get_text(config=config)


def chunk_text(
    text: str, chunk_size: int = 500, chunk_overlap: int = 50, doc_meta: DocumentMetadata | None = None
) -> list[Document]:
    """Chunk text into smaller pieces for embedding.

    Args:
        text (str): Text to chunk.
        chunk_size (int, optional): Size of the text chunk. Defaults to 500.
        chunk_overlap (int, optional): Size of the overlap with other chunks. Defaults to 50.
        doc_meta (DocumentMetadata | None, optional): Metadata on the document (url and title). Defaults to None.

    Returns:
        list[Document]: List of documents with the chunked text.
    """
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDINGS_MODEL)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).from_huggingface_tokenizer(tokenizer=tokenizer)

    docs = text_splitter.create_documents([text], metadatas=[doc_meta.model_dump()])
    logger.info(f"Created {len(docs)} splits")

    return docs


def embed_text(docs: list[LCDocument]) -> list[float]:
    """Embed text using a HuggingFace model.

    Args:
        docs (list[LCDocument]): List of documents to embed.

    Returns:
        list[float]: List of embeddings for the documents.
    """
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    docs = [split.page_content for split in docs]
    embeddings = embeddings_model.embed_documents(docs)
    logger.info(f"Created {len(embeddings)} embeddings")

    return embeddings


def save_embeddings(embeddings: list[str], docs: list[LCDocument]):
    """Save embeddings to a database.

    Args:
        embeddings (list[str]): List of embeddings to save.
        docs (list[LCDocument]): List of documents to save.
    """
    Base.metadata.create_all(engine)

    with Session(bind=engine) as session:
        for embedding, document in zip(embeddings, docs):
            doc = Document(
                embeddings=embedding,
                doc_title=document.metadata["title"].split("/")[-1],
                url=document.metadata["url"],
                content=document.page_content,
                tokens=len(document.page_content.split()),
                date_indexed=datetime.datetime.now(),
            )
            session.add(doc)
        session.commit()
    logger.info("Embeddings saved to database.")


if __name__ == "__main__":
    data_directory = pathlib.Path("data")
    counter = 0
    for pdf in tqdm.tqdm(list(data_directory.glob("*.pdf"))):
        text = text_extraction(pdf)
