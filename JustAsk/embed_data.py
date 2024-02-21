import torch
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer

class EmbedData:
    def __init__(self, text_model_name: str ="sentence-transformers/all-MiniLM-L12-v2", image_model_name:str = "openai/clip-vit-base-patch32"):
        # load text model
        self.text_model = SentenceTransformer(text_model_name)
        # load image model
        self.image_model = CLIPModel.from_pretrained(image_model_name)
        self.image_processor = CLIPProcessor.from_pretrained(image_model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_model.to(self.device)

    def get_text_embedding(self, text):
        return self.text_model.encode(text).tolist()

    @torch.no_grad()
    def get_query_embeddings_for_image(self, query):
        inputs = self.image_processor(text=query, return_tensors="pt")
        inputs = inputs.to(self.device)
        query_features = self.image_model.get_text_features(**inputs)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        return query_features.tolist()

    @torch.no_grad()
    def get_image_embeddings(self, images):
        inputs = self.image_processor(images=images, return_tensors="pt")
        inputs = inputs.to(self.device)
        image_embeddings = self.image_model.get_image_features(**inputs)
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        return image_embeddings.tolist()