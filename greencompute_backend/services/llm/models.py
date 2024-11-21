from typing import Literal

from pydantic import BaseModel


class RetrievalRequest(BaseModel):
    query: str
    top_n: int = 20
    format: bool = False


class DocumentResponse(BaseModel):
    doc_title: str
    content: str
    url: str


class RetrievalResponse(BaseModel):
    documents: list[DocumentResponse] | str


class LLMPrompt(BaseModel):
    body: str = "How can I increase my data center efficiency?"
    llm_id: str = "amazon.titan-text-premier-v1:0"
    max_tokens: int = 512
    stop_sequences: list = []
    temperature: float = 0.3
    top_p: float = 0.7
    top_n: int = 20
    prompt: Literal["base", "cite"]


class LLMResponse(BaseModel):
    response: str
    context: list[DocumentResponse]
    llm_id: str = "amazon.titan-text-premier-v1:0"
