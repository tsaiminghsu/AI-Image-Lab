"""Image -> prompt captioning for the GUI's "generate prompt from an uploaded
image" feature. Uses BLIP (base) via transformers - lazy-loaded on first use
so app startup doesn't pay the model-load cost. Runs on CPU deliberately:
ComfyUI already holds the SDXL checkpoint in this machine's 8GB VRAM, so
putting a second model on the GPU risks OOM-ing an in-progress generation.
"""

_processor = None
_model = None


def _load():
    global _processor, _model
    if _model is None:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        _model.eval()
    return _processor, _model


def caption_image(image_path: str) -> str:
    """Returns a short English caption describing the uploaded image, suitable
    to paste into (or edit into) the GUI's Prompt field."""
    from PIL import Image

    processor, model = _load()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True)
