import torch
from typing import List, Dict, Any
from transformers import CLIPTokenizer, CLIPProcessor
from PIL import Image
from pathlib import Path


class InferenceCollator:
    def __init__(self, tokenizer: CLIPTokenizer, processor: CLIPProcessor):
        self.tokenizer = tokenizer
        self.processor = processor
        self.default_image = Image.new('RGB', (224, 224), (0, 0, 0))

    def _load_image(self, img_path: Any) -> Image.Image:
        if not img_path or not isinstance(img_path, str):
            return self.default_image
        p = Path(str(img_path).replace("\\", "/"))
        if not p.exists():
            return self.default_image
        try:
            return Image.open(p).convert('RGB')
        except Exception:
            return self.default_image

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = []
        for item in batch:
            summary = str(item.get('summary') or "")
            name = str(item.get('short_name') or "")
            text = summary if len(summary.split()) > 3 else name
            texts.append(text if text else " ")

        tokenized = self.tokenizer(
            texts, padding='max_length', truncation=True, max_length=77, return_tensors='pt'
        )
        images = [self._load_image(item.get('image_path')) for item in batch]
        image_inputs = self.processor(images=images, return_tensors='pt')

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'pixel_values': image_inputs['pixel_values']
        }
