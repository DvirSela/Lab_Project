import torch
from typing import List, Tuple, Dict, Any
from pathlib import Path
from functools import lru_cache
from transformers import CLIPTokenizer, CLIPProcessor
from PIL import Image

class MultimodalCollator:
    """
    Custom collate function to process a batch of (anchor, positive) pairs.
    It tokenizes text and processes images for the *entire* 2*B batch at once.
    Optimized with image caching to reduce redundant I/O operations.
    
    Note: The image cache is per-instance. If multiple MultimodalCollator instances
    are created, each will have its own cache. For typical use cases with a single
    DataLoader, this is the desired behavior to avoid cache pollution across different
    datasets or configurations.
    """
    def __init__(self, tokenizer: CLIPTokenizer, processor: CLIPProcessor, cache_size: int = 128):
        self.tokenizer = tokenizer
        self.processor = processor
        self.default_image = Image.new('RGB', (224, 224), (0, 0, 0))
        # Use LRU cache for frequently accessed images (per-instance)
        self._load_image_cached = lru_cache(maxsize=cache_size)(self._load_image_uncached)

    def _load_image_uncached(self, img_path: str) -> Image.Image:
        """Loads an image without caching (used internally by cached version)."""
        if not img_path or not isinstance(img_path, str):
            return self.default_image
        
        p = Path(str(img_path).replace("\\", "/"))
        if not p.exists():
            return self.default_image
        try:
            return Image.open(p).convert('RGB')
        except Exception:
            return self.default_image

    def _load_image(self, img_path: Any) -> Image.Image:
        """Safely loads an image with caching, returning a default on failure."""
        if not img_path or not isinstance(img_path, str):
            return self.default_image
        return self._load_image_cached(img_path)

    def __call__(self, batch: List[Tuple[Dict, Dict]]) -> Dict[str, torch.Tensor]:
        # batch is a list of B (anchor_data, positive_data) tuples
        anchors = [item[0] for item in batch]
        positives = [item[1] for item in batch]
        all_items = anchors + positives # Total 2*B items
        
        # --- Process Texts ---
        texts = []
        for item in all_items:
            summary = str(item.get('summary') or "")
            name = str(item.get('short_name') or "")
            # Use summary if it's substantial, otherwise fallback to name
            text = summary if len(summary.split()) > 3 else name
            if not text:
                text = " " # Use a single space if both are empty
            texts.append(text)
            
        tokenized = self.tokenizer(
            texts, 
            padding='max_length',  # Pad to model max length
            truncation=True, 
            max_length=77, 
            return_tensors='pt'
        )
        
        # --- Process Images ---
        images = [self._load_image(item.get('image_path')) for item in all_items]
        image_inputs = self.processor(
            images=images, 
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'pixel_values': image_inputs['pixel_values']
        }