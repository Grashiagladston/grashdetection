from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model once
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("clip-ViT-B-32")
    return _model


def generate_embedding(pil_image):

    if not isinstance(pil_image, Image.Image):
        pil_image = Image.open(pil_image)

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    model = get_model()

    embedding = model.encode(
        pil_image,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding.astype(np.float32)
