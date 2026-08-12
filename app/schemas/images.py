from pydantic import BaseModel, Field, field_validator


class ImageGenerationRequest(BaseModel):
    model: str = Field(description="Configured Qwen-Image model ID")
    prompt: str = Field(
        min_length=1,
        max_length=10_000,
        description="Text prompt used to generate the image",
    )
    size: str = Field(
        default="1024x1024",
        description="Output dimensions in WIDTHxHEIGHT format",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description="Optional random seed for deterministic generation",
    )
    negative_prompt: str | None = Field(
        default=None,
        max_length=10_000,
        description="Optional text describing content to avoid",
    )
    num_inference_steps: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of denoising steps; defaults to the configured Lightning value",
    )
    true_cfg_scale: float | None = Field(
        default=None,
        ge=1.0,
        le=20.0,
        description="Classifier-free guidance scale; 1.0 disables true CFG",
    )

    @field_validator("size")
    @classmethod
    def validate_size(cls, value: str) -> str:
        parts = value.split("x")
        if len(parts) != 2 or any(not part.isdigit() for part in parts):
            raise ValueError("size must use WIDTHxHEIGHT format")
        width, height = (int(part) for part in parts)
        if width < 256 or height < 256:
            raise ValueError("width and height must be at least 256")
        return value


class ImageData(BaseModel):
    b64_json: str


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]
