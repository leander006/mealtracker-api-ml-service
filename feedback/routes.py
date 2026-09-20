from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session, sessionmaker

from feedback.cloudinary_client import upload_correction_image
from feedback.db_router import TrainingDBRouter
from feedback.models import Correction

router = APIRouter(prefix="/feedback", tags=["feedback"])


def get_training_router(request) -> TrainingDBRouter:
    # Set on app.state at startup in main.py - see main.py's lifespan handler.
    return request.app.state.training_db_router


@router.post("/corrections")
async def submit_correction(
    image: UploadFile = File(...),
    predicted_label: str | None = Form(None),
    predicted_confidence: float | None = Form(None),
    corrected_label: str = Form(...),
    corrected_weight_g: float | None = Form(None),
    was_low_confidence_flow: bool = Form(False),
    user_id: int | None = Form(None),
    training_router: TrainingDBRouter = Depends(get_training_router),
):
    image_bytes = await image.read()
    image_url = upload_correction_image(image_bytes, user_id)

    engine = training_router.get_write_engine()
    Session = sessionmaker(bind=engine)
    with Session() as session:
        correction = Correction(
            user_id=user_id,
            image_url=image_url,
            predicted_label=predicted_label,
            predicted_confidence=predicted_confidence,
            corrected_label=corrected_label,
            corrected_weight_g=corrected_weight_g,
            was_low_confidence_flow=was_low_confidence_flow,
        )
        session.add(correction)
        session.commit()
        return {"id": correction.id, "image_url": image_url}
