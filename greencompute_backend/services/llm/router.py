from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from greencompute_backend.resources import get_bedrock_client, get_db

from .models import LLMPrompt, LLMResponse, RetrievalRequest, RetrievalResponse
from .svc import PROMPT, fmt_docs, prompt_bedrock, retrieve_docs

router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("/prompt", response_model=LLMResponse)
def bedrock(prompt: LLMPrompt, client=Depends(get_bedrock_client)):
    return prompt_bedrock(prompt, client)


@router.post("/rag", response_model=LLMResponse)
def rag(
    prompt: LLMPrompt,
    bedrock_client=Depends(get_bedrock_client),
    db_client=Depends(get_db),
):
    docs = retrieve_docs(prompt.body, top_k=10, db=db_client)
    docs_formatted = fmt_docs(docs)
    prompt.body = PROMPT.format(context=docs_formatted, question=prompt.body)
    bedrock_response = prompt_bedrock(prompt, bedrock_client)

    return LLMResponse(context=docs, **bedrock_response)


@router.post("/retrieval", response_model=RetrievalResponse)
async def retrieve_documents(request: RetrievalRequest, db: Session = Depends(get_db)):
    results = retrieve_docs(request.query, request.top_k, db)
    return {"documents": results}
