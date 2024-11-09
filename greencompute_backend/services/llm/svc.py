import json
import pathlib
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import select
from sqlalchemy.orm import Session

from greencompute_backend.db.tables import Document
from greencompute_backend.resources import get_db
from greencompute_backend.services.llm.models import DocumentResponse, LLMPrompt

EMBEDDINGS_MODEL = "BAAI/bge-base-en-v1.5"
KEYS = ["doc_title", "content", "url"]
PROMPT = """
Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.
Context:
{context}
-----------------------------------
Now here is the question you need to answer.

Question:
{question}
"""

embeddings = {}


def embed(query: str):
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    return embeddings_model.embed_query(query)


@asynccontextmanager
async def lifespan_embeddings(app: FastAPI):
    # Load the embeddings model
    embeddings[EMBEDDINGS_MODEL] = embed
    yield
    # Clean up the embeddings model and release the resources
    embeddings.clear()


def prompt_bedrock(prompt: LLMPrompt, client: object) -> dict[str, str]:
    """Create a payload for the bedrock client and return the response.

    Args:
        prompt (LLMPrompt): Prompt object.
        client (object): Bedrock client.

    Returns:
        dict[str, str]: Response from the bedrock client.
    """
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
        "response": response_body.get("results")[0].get("outputText"),
        "llm_id": prompt.llm_id,
    }


def stream_prompt_bedrock(prompt: LLMPrompt, client: object):
    """Create a payload for the bedrock client and return the response.

    Args:
        prompt (LLMPrompt): Prompt object.
        client (object): Bedrock client.

    Returns:
        LLMResponse: Response from the bedrock client.
    """
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
    # Invoke the model with the request.
    streaming_response = client.invoke_model_with_response_stream(modelId=prompt.llm_id, body=body)

    # Extract and print the response text in real-time.
    for event in streaming_response["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        if "outputText" in chunk:
            yield chunk["outputText"]


async def retrieve_docs(query: str, top_k: int = 10, db: Session = Depends(get_db)) -> list[DocumentResponse]:
    """Document retrieval based on embedded query.

    Args:
            query (str): Query to search for.
            top_k (int, optional): Number of top documents to return. Defaults to 10.
            db (Session, optional): Document session . Defaults to Depends(get_db).

    Returns:
            list[DocumentResponse]: List of documents.
    """
    q_embed = embeddings[EMBEDDINGS_MODEL](query)
    result = db.scalars(select(Document).order_by(Document.embeddings.l2_distance(q_embed).desc()).limit(top_k))
    results = [r.__dict__ for r in result.fetchall()]
    results = [{k: v for k, v in r.items() if k in KEYS} for r in results]
    results = [DocumentResponse(**r) for r in results]
    return results


def fmt_docs(docs: list[DocumentResponse]):
    context = ""
    for i, doc in enumerate(docs):
        context += f"[{i + 1}]. [url: {doc.url}]. [title: {doc.doc_title}]\n{doc.content}\n\n"
    return context


def select_prompt(prompt: Literal["base", "cite"]) -> str:
    _dir = pathlib.Path(__file__).parent
    with open(f"{str(_dir)}/prompts/{prompt}.txt", "r") as f:
        return f.read()


if __name__ == "__main__":
    import pathlib

    print(pathlib.Path(__file__))
