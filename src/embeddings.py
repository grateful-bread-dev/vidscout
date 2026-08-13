from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


DEFAULT_MODEL = "openai/clip-vit-base-patch32"


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def extract_feature_tensor(output) -> torch.Tensor:
    """
    Extract the projected CLIP feature tensor from the model output.

    Recent Transformers versions return a BaseModelOutputWithPooling
    object from get_image_features() and get_text_features(), with the
    projected feature stored in pooler_output.

    Older versions may return the tensor directly.
    """
    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "pooler_output"):
        return output.pooler_output

    raise TypeError(
        "Unexpected CLIP feature output type: "
        f"{type(output).__name__}"
    )


class ClipSearchEngine:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.device = choose_device()

        print(f"Loading CLIP on: {self.device}")

        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model = CLIPModel.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

    def embed_images(
        self,
        image_paths: list[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        all_embeddings = []

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]

            images = []

            for path in batch_paths:
                with Image.open(Path(path)) as image:
                    images.append(image.convert("RGB"))

            inputs = self.processor(
                images=images,
                return_tensors="pt",
            )

            pixel_values = inputs["pixel_values"].to(self.device)

            with torch.inference_mode():
                outputs = self.model.get_image_features(
                    pixel_values=pixel_values
                )

                embeddings = extract_feature_tensor(outputs)

                embeddings = F.normalize(
                    embeddings,
                    p=2,
                    dim=-1,
                )

            all_embeddings.append(
                embeddings.cpu().numpy()
            )

        return np.vstack(all_embeddings)

    def embed_text(self, query: str) -> np.ndarray:
        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True,
        )

        model_inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask"}
        }

        with torch.inference_mode():
            outputs = self.model.get_text_features(
                **model_inputs
            )

            embedding = extract_feature_tensor(outputs)

            embedding = F.normalize(
                embedding,
                p=2,
                dim=-1,
            )

        return embedding.cpu().numpy()[0]