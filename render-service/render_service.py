from __future__ import annotations
import asyncio
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from loguru import logger
import torch

import render
from pathlib import Path

MAX_UPLOAD_SIZE = 200 * 1024 * 1024

app = FastAPI(title="Render Service", version="1.0.0")


# Warmup: trigger render module lazy initialization after first startup
async def _warmup_render() -> None:
    logger.info("Render warmup: starting")
    await asyncio.sleep(0.5)
    loop = asyncio.get_event_loop()
    # Look for a warmup.glb next to this script (optional)
    warmup_path = Path(__file__).parent / "warmup.glb"
    if not warmup_path.exists():
        logger.warning(f"Warmup GLB not found at {warmup_path}; skipping warmup")
        return

    try:
        # read warmup file bytes and invoke render to trigger lazy initialization
        payload = warmup_path.read_bytes()
        await loop.run_in_executor(None, lambda: render.grid_from_glb_bytes(payload))
        logger.info("Render warmup: completed successfully")
    except Exception as exc:
        # ignore errors from payload/render while still forcing initialization
        logger.warning(f"Render warmup: finished with exception (ignored): {exc}")


@app.on_event("startup")
async def _on_startup() -> None:
    asyncio.create_task(_warmup_render())


def _resolve_device(preferred: str | None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

async def _read_upload_with_limit(file: UploadFile, max_size: int) -> bytes:
    """Read upload file with size limit, abort early if exceeded."""
    chunks = []
    total_size = 0
    chunk_size = 64 * 1024  # 64KB chunks
    
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB"
            )
        chunks.append(chunk)
    
    return b"".join(chunks)

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/render_ply")
async def render_ply(
    file: UploadFile = File(...),
    device: Literal["cuda", "cpu"] | None = None,
) -> Response:
    filename = file.filename or ""
    if not filename.lower().endswith(".ply"):
        raise HTTPException(status_code=400, detail="Only .ply files are supported")

    payload = await _read_upload_with_limit(file, MAX_UPLOAD_SIZE)
    torch_device = _resolve_device(device)

    try:
        png_bytes = await asyncio.get_event_loop().run_in_executor(
            None, lambda: render.grid_from_ply_bytes(payload, torch_device)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to render uploaded file: {exc}"
        ) from exc

    return Response(content=png_bytes, media_type="image/png")

@app.post("/render_glb")
async def render_glb(
    file: UploadFile = File(...),
    device: Literal["cuda", "cpu"] | None = None,
) -> Response:
    filename = file.filename or ""
    logger.info(f"render_glb request received: filename={filename!r}")
    
    if not filename.lower().endswith(".glb"):
        logger.warning(f"Invalid file extension rejected: {filename!r}")
        raise HTTPException(status_code=400, detail="Only .glb files are supported on /render_glb")

    payload = await _read_upload_with_limit(file, MAX_UPLOAD_SIZE)
    logger.info(f"File uploaded: {len(payload)} bytes")
    
    torch_device = _resolve_device(device)
    logger.debug(f"Using device: {torch_device}")

    try:
        logger.info("Starting GLB render...")
        png_bytes = await asyncio.get_event_loop().run_in_executor(
            None, lambda: render.grid_from_glb_bytes(payload)
        )
        logger.info(f"Render complete, returning {len(png_bytes)} bytes")
    except ValueError as exc:
        logger.error(f"ValueError during render: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"Exception during render: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Failed to render uploaded file: {exc}"
        ) from exc

    return Response(content=png_bytes, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)