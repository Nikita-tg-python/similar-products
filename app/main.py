from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app import health
from app.config import settings

app = FastAPI(title="Similar Products", version="0.1.0")


@app.get("/health")
async def get_health() -> JSONResponse:
    services = await health.check_all(settings)
    all_ok = all(services.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={
            "status": "ok" if all_ok else "degraded",
            **{name: "ok" if ok else "error" for name, ok in services.items()},
            "llm_provider": settings.llm_provider,
        },
    )
