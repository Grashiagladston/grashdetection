from PIL import Image
import numpy as np
import torch
import open_clip

# ============================================================
# GLOBAL MODEL CACHE
# ============================================================

_model = None
_preprocess = None
_device = None


def _load_model():
    """
    Load OpenCLIP model only once.
    """

    global _model
    global _preprocess
    global _device

    if _model is None:

        _device = "cuda" if torch.cuda.is_available() else "cpu"

        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k"
        )

        _model.to(_device)
        _model.eval()


# ============================================================
# IMAGE EMBEDDING
# ============================================================

def generate_embedding(pil_image):
    print("TEST EMBEDDING CALLED")
    return [0.0] * 512

    """
    Generate a normalized CLIP embedding.

    Returns:
        list[float]
    """

    _load_model()

    if not isinstance(pil_image, Image.Image):
        pil_image = Image.open(pil_image)

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    image_tensor = _preprocess(pil_image).unsqueeze(0).to(_device)

    with torch.no_grad():

        embedding = _model.encode_image(image_tensor)

        embedding /= embedding.norm(dim=-1, keepdim=True)

    embedding = embedding.squeeze().cpu().numpy()

    return embedding.astype(np.float32).tolist()
