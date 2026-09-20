"""Thin wrapper around Cloudinary's upload API. Kept as one function so it's
trivial to swap providers later without touching feedback/routes.py.
"""
from __future__ import annotations

import cloudinary
import cloudinary.uploader

from config import settings

cloudinary.config(
    cloud_name=settings.cloudinary_cloud_name,
    api_key=settings.cloudinary_api_key,
    api_secret=settings.cloudinary_api_secret,
    secure=True,
)


def upload_correction_image(image_bytes: bytes, user_id: int | None) -> str:
    """Uploads an image and returns its public URL. Tags with the user id
    (when available) and a fixed folder so you can browse/audit corrections
    in the Cloudinary console without querying the DB."""
    result = cloudinary.uploader.upload(
        image_bytes,
        folder="meal-corrections",
        tags=[f"user:{user_id}"] if user_id else ["user:anonymous"],
    )
    return result["secure_url"]
