from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L12-v2")

class MatchRequest(BaseModel):
    texts: list[str]  # Example inputs
    query: str         # User query string

class MatchResponse(BaseModel):
    index: int
    score: float

@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    # Encode all input texts and the query
    input_embeddings = model.encode(req.texts, convert_to_tensor=True)
    query_embedding = model.encode(req.query, convert_to_tensor=True)

    # Compute cosine similarity
    scores = util.cos_sim(input_embeddings, query_embedding)[..., 0]  # shape: (len(texts),)

    best_index = torch.argmax(scores).item()
    best_score = scores[best_index].item()

    return MatchResponse(index=best_index, score=best_score)