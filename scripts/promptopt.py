import json

import dspy
import pandas as pd
import requests
from dspy.evaluate import SemanticF1


def create_dataset(path: str):
    # Read in synthetic datset
    dataset = pd.read_excel(path)
    # create the dspy dataset
    dataset_dict = dataset.to_dict(orient="records")
    dspy_dataset = []

    for row in dataset_dict:
        dspy_dataset.append(
            dspy.Example(question=row["input"], response=row["expected_output"]).with_inputs("question")
        )

    return dspy_dataset


def create_sets(dataset: list[dspy.Example], metric: dspy.Module = SemanticF1()):
    trainset, valset, devset, testset = dataset[:10], dataset[10:20], dataset[20:30], dataset[30:40]
    evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24, display_progress=True, display_table=3)

    return trainset, valset, devset, testset, evaluate


def search(query: str, top_k: int) -> list[str]:
    url = "http://greencompute-1575332443.us-east-1.elb.amazonaws.com/api/llm/retrieval"
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    data = {"query": query, "top_k": top_k}

    documents = requests.post(url, headers=headers, json=data).json()["documents"]
    return [doc["doc_title"] + doc["url"] + "\n\n" + doc["content"] for doc in documents]


class TitanLM(dspy.LM):
    def __init__(
        self, model: str, client, max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.9, **kwargs
    ):
        self.client = client
        self.history = []
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        super().__init__(model, **kwargs)
        self.model = model

    def _format_message(self, prompt: str):
        body = json.dumps(
            {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": self.max_tokens,
                    "stopSequences": [],
                    "temperature": self.temperature,
                    "topP": self.top_p,
                },
            }
        )
        return body

    def generate_content(self, prompt: str) -> str:
        body = self._format_message(prompt)
        response = self.client.invoke_model(
            body=body,
            modelId=self.model,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("results")

    def __call__(self, prompt=None, messages=None, **kwargs):
        # Custom chat model working for text completion model
        prompt = "\n\n".join([x["content"] for x in messages] + ["BEGIN RESPONSE:"])

        completions = self.generate_content(prompt)
        self.history.append({"prompt": prompt, "completions": completions})

        # Must return a list of strings
        return [completions[0].get("outputText")]

    def inspect_history(self):
        for interaction in self.history:
            print(f"Prompt: {interaction['prompt']} -> Completions: {interaction['completions']}")


class GenerateCitedParagraph(dspy.Signature):
    """Generate a paragraph with citations."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    paragraph = dspy.OutputField(desc="includes citations with links")


class RAG(dspy.Module):
    def __init__(self, num_docs=20):
        self.num_docs = num_docs
        self.respond = dspy.ChainOfThought(GenerateCitedParagraph)

    def forward(self, question):
        context = search(question, top_k=self.num_docs)
        return self.respond(context=context, question=question)
