from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from greencompute_backend.resources import get_bedrock_client, get_db

from .models import LLMPrompt, LLMResponse, RetrievalRequest, RetrievalResponse
from .svc import (
    fmt_docs,
    lifespan_embeddings,
    prompt_bedrock,
    retrieve_docs,
    select_prompt,
    stream_prompt_bedrock,
)

router = APIRouter(prefix="/llm", tags=["llm"], lifespan=lifespan_embeddings)


@router.post("/prompt")
def bedrock(prompt: LLMPrompt, client=Depends(get_bedrock_client)) -> dict[str, str]:
    return prompt_bedrock(prompt, client)


@router.post("/retrieval", response_model=RetrievalResponse)
async def retrieve_documents(request: RetrievalRequest, db: Session = Depends(get_db)):
    results = await retrieve_docs(request.query, request.top_k, db)
    if request.format:
        return {"documents": fmt_docs(results)}
    return {"documents": results}


@router.post("/rag", response_model=LLMResponse)
async def rag(
    prompt: LLMPrompt,
    bedrock_client=Depends(get_bedrock_client),
    db_client=Depends(get_db),
):
    docs = await retrieve_docs(prompt.body, top_k=prompt.top_k, db=db_client)
    docs_formatted = fmt_docs(docs)
    prompt.body = select_prompt(prompt.prompt).format(context=docs_formatted, question=prompt.body)
    bedrock_response = prompt_bedrock(prompt, bedrock_client)

    return LLMResponse(context=docs, **bedrock_response)


@router.post("/stream-rag", response_model=LLMResponse)
async def stream_rag(prompt: LLMPrompt, client=Depends(get_bedrock_client), db=Depends(get_db)):
    docs = await retrieve_docs(prompt.body, top_k=prompt.top_k, db=db)
    docs_formatted = fmt_docs(docs)
    prompt.body = select_prompt(prompt.prompt).format(context=docs_formatted, question=prompt.body)
    return StreamingResponse(stream_prompt_bedrock(prompt, client))
