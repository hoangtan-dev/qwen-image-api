# Qwen Image Server

OpenAI-compatible FastAPI server for Qwen-Image-2512

## Development

```bash
uv sync
cp .env.example .env
uv run uvicorn app.main:app
```

OpenAI-compatible image generation:

```bash
curl http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen-Image-2512","prompt":"A cinematic mountain landscape at sunrise","size":"1024x1024","response_format":"b64_json"}'
```

The server loads `Qwen/Qwen-Image-2512`, quantizes its transformer to FP8 with TorchAO, and loads the four-step Lightning LoRA by default.
