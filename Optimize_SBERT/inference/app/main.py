from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import torch
from optimum.onnxruntime import ORTModelForFeatureExtraction
import os

app = FastAPI()

optimized_model_path = os.getenv('OPTIMIZED_MODEL_PATH')

# Load a pre-trained optimized SBERT model
tokenizer = AutoTokenizer.from_pretrained(optimized_model_path, model_max_length=512)

model = ORTModelForFeatureExtraction.from_pretrained(optimized_model_path)

class EmbeddingInput(BaseModel):
    text: str

class EmbeddingOutput(BaseModel):
    embedding: list

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def compute_embedding(tokenizer, model, sentence):

    # Tokenize sentences
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    return sentence_embeddings.tolist()


@app.post("/generate_embedding", response_model=EmbeddingOutput)
async def generate_embedding(input_data: EmbeddingInput):
    # Generate embedding for the input text
    embedding = compute_embedding(tokenizer = tokenizer, model = model, sentence=input_data.text)
    return {"embedding": embedding}