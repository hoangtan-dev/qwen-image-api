from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    debug: bool = False
    app_env: str = "local"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    model_id: str = "Qwen/Qwen-Image-2512"
    lightning_model_id: str = "lightx2v/Qwen-Image-2512-Lightning"
    lightning_weight_name: str = "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    quantize_transformer_fp8: bool = True
    compile_model: bool = True
    warmup_model: bool = True
    load_model_on_startup: bool = True
    default_num_inference_steps: int = 4
    default_true_cfg_scale: float = 1.0
    max_width: int = 1664
    max_height: int = 1664


settings = Settings()
