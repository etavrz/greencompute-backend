import json

from fastapi import APIRouter, Depends
from loguru import logger
from pydantic import BaseModel

from greencompute_backend.resources import get_bedrock_client, get_db
from greencompute_backend.routes.retrieval import retrieve_docs

router = APIRouter(prefix="/llm", tags=["llm"])


class LLMPrompt(BaseModel):
    body: str = "tell me a joke"
    llm_id: str = "amazon.titan-text-premier-v1:0"
    max_tokens: int = 256
    stop_sequences: list = []
    temperature: float = 0
    top_p: float = 0.9


class LLMResponse(BaseModel):
    body: str
    llm_id: str = "amazon.titan-text-premier-v1:0"


def prompt_bedrock(prompt: LLMPrompt, client=Depends(get_bedrock_client)):
    body = json.dumps(
        {
            "inputText": prompt.body,
            "textGenerationConfig": {
                "maxTokenCount": prompt.max_tokens,
                "stopSequences": prompt.stop_sequences,
                "temperature": prompt.temperature,
                "topP": prompt.top_p,
            },
        }
    )
    response = client.invoke_model(
        body=body,
        modelId=prompt.llm_id,
        accept="application/json",
        contentType="application/json",
    )
    response_body = json.loads(response.get("body").read())
    return {
        "body": response_body.get("results")[0].get("outputText"),
        "llm_id": prompt.llm_id,
    }


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
