from pydantic import BaseModel


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


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 10


class DocumentResponse(BaseModel):
    doc_title: str
    content: str


class RetrievalResponse(BaseModel):
    documents: list[DocumentResponse]
