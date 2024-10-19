from pydantic import BaseModel


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 10


class DocumentResponse(BaseModel):
    doc_title: str
    content: str


class RetrievalResponse(BaseModel):
    documents: list[DocumentResponse]


class LLMPrompt(BaseModel):
    body: str = "How can I increase my data center efficiency?"
    llm_id: str = "amazon.titan-text-premier-v1:0"
    max_tokens: int = 512
    stop_sequences: list = []
    temperature: float = 0.7
    top_p: float = 0.9


class LLMResponse(BaseModel):
    response: str
    context: list[DocumentResponse]
    llm_id: str = "amazon.titan-text-premier-v1:0"
