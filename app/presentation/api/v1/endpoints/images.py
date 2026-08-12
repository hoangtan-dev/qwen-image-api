import time

from dishka import FromDishka
from dishka.integrations.fastapi import inject
from fastapi import APIRouter, HTTPException

from app.core.config import Settings
from app.schemas.images import ImageData, ImageGenerationRequest, ImageGenerationResponse
from app.services.image_generation import ImageGenerationService

router = APIRouter(prefix="/v1", tags=["Images"])


@router.post("/images/generations", response_model=ImageGenerationResponse)
@inject
async def generate_image(
    request: ImageGenerationRequest,
    service: FromDishka[ImageGenerationService],
    settings: FromDishka[Settings],
) -> ImageGenerationResponse:
    if request.model != settings.model_id:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "The requested model is not available", "type": "invalid_request_error"}},
        )
    width, height = (int(value) for value in request.size.split("x"))
    if width > settings.max_width or height > settings.max_height:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "Requested image size exceeds the configured limit",
                    "type": "invalid_request_error",
                }
            },
        )
    try:
        image = await service.generate_async(
            request.prompt,
            negative_prompt=request.negative_prompt,
            width=width,
            height=height,
            seed=request.seed,
            num_inference_steps=request.num_inference_steps,
            true_cfg_scale=request.true_cfg_scale,
        )
    except RuntimeError as error:
        raise HTTPException(
            status_code=503, detail={"error": {"message": str(error), "type": "server_error"}}
        ) from error
    return ImageGenerationResponse(created=int(time.time()), data=[ImageData(b64_json=image.b64_json)])
