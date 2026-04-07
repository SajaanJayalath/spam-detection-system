from typing import Dict, Any
from fastapi import Request
from fastapi.responses import JSONResponse


class ModelNotLoadedError(RuntimeError):
    pass


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "message": "Invalid input. Please provide a valid message.",
        },
    )


async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError) -> JSONResponse:  # noqa: ARG001
    return JSONResponse(
        status_code=500,
        content={
            "error": "ModelNotLoaded",
            "message": "Model artifacts are not loaded. Ensure training is complete and files are present.",
        },
    )
