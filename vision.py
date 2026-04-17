"""Vision captioning via Ollama multimodal models (e.g. qwen2.5vl:3b).

SVG files are returned as-is (already text). Raster images (PNG, JPG, etc.)
are sent to Ollama's /api/generate with base64-encoded image data.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image

_CAPTION_PROMPT = (
    "Describe this image in detail. Focus on architecture, data flow, technical content, "
    "labels, and any text visible in the image. If this is a diagram, describe the "
    "components and their relationships."
)

_SVG_EXTENSIONS = frozenset({".svg"})
_RASTER_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
_ALL_IMAGE_EXTENSIONS = _SVG_EXTENSIONS | _RASTER_EXTENSIONS


def get_image_metadata(path: Path) -> Optional[dict]:
    """Return metadata dict for an image file, or None if the file cannot be read.

    SVG files are treated as text (is_text=True, width=0, height=0).
    Raster images are opened with Pillow to extract width, height, and format.
    """
    if not path.exists():
        return None

    suffix = path.suffix.lower()

    if suffix in _SVG_EXTENSIONS:
        return {
            "format": "SVG",
            "width": 0,
            "height": 0,
            "file_size": path.stat().st_size,
            "is_text": True,
        }

    try:
        with Image.open(path) as img:
            width, height = img.size
            fmt = img.format or suffix.lstrip(".").upper()
        return {
            "format": fmt,
            "width": width,
            "height": height,
            "file_size": path.stat().st_size,
            "is_text": False,
        }
    except Exception:
        return None


def caption_image(
    path: Path,
    ollama_url: str,
    model: str = "qwen2.5vl:3b",
    timeout: float = 120.0,
) -> str:
    """Generate a text caption for a single image file.

    - SVG: returns the raw file text without calling Ollama.
    - Raster: base64-encodes the image and POSTs to Ollama's /api/generate.
    Returns an empty string on any error.
    """
    suffix = path.suffix.lower()

    if suffix in _SVG_EXTENSIONS:
        return path.read_text(encoding="utf-8")

    try:
        raw_bytes = path.read_bytes()
        b64_data = base64.b64encode(raw_bytes).decode("ascii")

        from src.ollama_client import generate as _oc_generate
        return _oc_generate(
            ollama_url, model, _CAPTION_PROMPT,
            images=[b64_data],
            stream=False,
            timeout=timeout,
        )

    except Exception:
        return ""


def caption_images_batch(
    paths: list[Path],
    ollama_url: str,
    model: str = "qwen2.5vl:3b",
) -> list[dict]:
    """Caption multiple image files.

    Returns a list of dicts, each with keys: path (str), caption (str), metadata (dict|None).
    """
    results = []
    for p in paths:
        caption = caption_image(p, ollama_url=ollama_url, model=model)
        metadata = get_image_metadata(p)
        results.append({"path": str(p), "caption": caption, "metadata": metadata})
    return results
