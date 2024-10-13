from fastapi import APIRouter, Depends
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from greencompute_backend.db.tables import Document
from greencompute_backend.resources import get_db

EMBEDDINGS_MODEL = "BAAI/bge-base-en-v1.5"
KEYS = ["doc_title", "content"]

embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 10


class DocumentResponse(BaseModel):
    doc_title: str
    content: str


class RetrievalResponse(BaseModel):
    documents: list[DocumentResponse]


def retrieve_docs(query: str, top_k: int = 10, db: Session = Depends(get_db)) -> dict[str, str]:
    """Document retrieval based on embedded query.

    Args:
            query (str): Query to search for.
            top_k (int, optional): Number of top documents to return. Defaults to 10.
            db (Session, optional): Document session . Defaults to Depends(get_db).

    Returns:
            dict[str, str]: List of documents.
    """
    q_embed = embeddings_model.embed_query(query)
    result = db.scalars(select(Document).order_by(Document.embeddings.l2_distance(q_embed).desc()).limit(top_k))
    results = [r.__dict__ for r in result.fetchall()]
    results = [{k: v for k, v in r.items() if k in KEYS} for r in results]
    return results


@router.post("/retrieval", response_model=RetrievalResponse)
async def retrieve_documents(request: RetrievalRequest, db: Session = Depends(get_db)):
    results = retrieve_docs(request.query, request.top_k, db)
    return {"documents": results}
