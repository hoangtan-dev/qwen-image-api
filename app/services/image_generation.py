import asyncio
import base64
import io
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from app.core.config import Settings


@dataclass(frozen=True)
class GeneratedImage:
    b64_json: str


class ImageGenerationService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipeline: Any = None
        self._lock = asyncio.Lock()
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        started_at = time.perf_counter()
        logger.info("Loading image pipeline dependencies")
        import torch
        from diffusers import DiffusionPipeline, PipelineQuantizationConfig, TorchAoConfig

        dtype = getattr(torch, self._settings.torch_dtype)
        logger.info(
            "Preparing pipeline load: model_id={}, device={}, dtype={}, fp8={}, compile={}",
            self._settings.model_id,
            self._settings.device,
            self._settings.torch_dtype,
            self._settings.quantize_transformer_fp8,
            self._settings.compile_model,
        )
        load_options: dict[str, Any] = {"torch_dtype": dtype}
        if self._settings.quantize_transformer_fp8:
            logger.info("Configuring FP8 weight-only quantization for transformer")
            from torchao.quantization import Float8WeightOnlyConfig

            quant_mapping: dict[str, Any] = {"transformer": TorchAoConfig(Float8WeightOnlyConfig())}
            load_options["quantization_config"] = PipelineQuantizationConfig(quant_mapping=quant_mapping)

        logger.info("Loading base Diffusers pipeline: {}", self._settings.model_id)
        pipeline = DiffusionPipeline.from_pretrained(self._settings.model_id, **load_options)
        logger.info("Moving pipeline to device: {}", self._settings.device)
        pipeline = pipeline.to(self._settings.device)
        logger.info(
            "Loading Lightning LoRA: model_id={}, weight_name={}",
            self._settings.lightning_model_id,
            self._settings.lightning_weight_name,
        )
        pipeline.load_lora_weights(
            self._settings.lightning_model_id,
            weight_name=self._settings.lightning_weight_name,
        )
        if self._settings.compile_model:
            logger.info("Compiling transformer with torch.compile")
            pipeline.transformer = torch.compile(pipeline.transformer)
        self._pipeline = pipeline
        logger.info("Image pipeline loaded in {:.2f}s", time.perf_counter() - started_at)

    def warmup(self) -> None:
        if self._pipeline is None:
            raise RuntimeError("Image pipeline is not loaded")
        if self._settings.warmup_model:
            started_at = time.perf_counter()
            logger.info("Starting image pipeline warmup")
            self.generate("warmup", width=1024, height=1024, seed=0)
            logger.info("Image pipeline warmup completed in {:.2f}s", time.perf_counter() - started_at)
        self._ready = True
        logger.info("Image generation service is ready")

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        true_cfg_scale: float | None = None,
    ) -> GeneratedImage:
        if self._pipeline is None:
            raise RuntimeError("Image pipeline is not loaded")

        import torch

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._settings.device).manual_seed(seed)
        parameters = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps or self._settings.default_num_inference_steps,
            "true_cfg_scale": true_cfg_scale or self._settings.default_true_cfg_scale,
            "generator": generator,
        }
        if negative_prompt is not None:
            parameters["negative_prompt"] = negative_prompt
        with torch.inference_mode():
            result = self._pipeline(**parameters).images[0]
        output = io.BytesIO()
        result.save(output, format="PNG")
        return GeneratedImage(base64.b64encode(output.getvalue()).decode("ascii"))

    async def generate_async(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        true_cfg_scale: float | None = None,
    ) -> GeneratedImage:
        async with self._lock:
            return await asyncio.to_thread(
                self.generate,
                prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                seed=seed,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
            )
