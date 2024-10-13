from fastapi import APIRouter, Depends
from loguru import logger
from sqlalchemy.orm import Session

from greencompute_backend.resources import get_bedrock_client, get_db

from .models import LLMPrompt, LLMResponse, RetrievalRequest, RetrievalResponse
from .svc import fmt_docs, prompt_bedrock, retrieve_docs

router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("/bedrock", response_model=LLMResponse)
def bedrock(prompt: LLMPrompt, client=Depends(get_bedrock_client)):
    return prompt_bedrock(prompt, client)


@router.post("/prompt", response_model=LLMResponse)
def rag(
    prompt: LLMPrompt,
    bedrock_client=Depends(get_bedrock_client),
    db_client=Depends(get_db),
):
    docs = retrieve_docs(prompt.body, top_k=10, db=Depends(db_client))
    logger.debug(docs)
    return prompt_bedrock(prompt, bedrock_client)


@router.post("/retrieval", response_model=RetrievalResponse)
async def retrieve_documents(request: RetrievalRequest, db: Session = Depends(get_db)):
    results = retrieve_docs(request.query, request.top_k, db)
    return {"documents": results}


@router.post("/format")
async def format_documents(request: RetrievalRequest, db: Session = Depends(get_db)):
    results = retrieve_docs(request.query, request.top_k, db)
    results = fmt_docs(results)
    return {"formatted_documents": results}
