"""FastAPI service implementing API_CONTRACT.md: GET /health and POST /ask."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

load_dotenv()

from .generator import ClaudeGenerator, GenerationError  # noqa: E402
from .ingest import load_corpus  # noqa: E402
from .qa import PolicyQA  # noqa: E402
from .retrieval import HybridRetriever  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("policy_qa")

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = Path(os.getenv("POLICY_DIR", ROOT / "data" / "policies"))
CACHE_DIR = os.getenv("FASTEMBED_CACHE", str(ROOT / ".cache" / "fastembed"))


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


def build_qa(generator=None) -> PolicyQA:
    chunks, _ = load_corpus(POLICY_DIR)
    retriever = HybridRetriever(chunks, cache_dir=CACHE_DIR)
    log.info("indexed %d sections from %s", len(chunks), POLICY_DIR)
    return PolicyQA(retriever, generator or ClaudeGenerator(), chunks)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "qa", None) is None:  # tests may inject their own
        app.state.qa = build_qa()
    yield


app = FastAPI(title="Kestrel Nova Policy Q&A", lifespan=lifespan)


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    msg = first.get("msg", "invalid request")
    return error(422, f"{field}: {msg}" if field else msg)


@app.exception_handler(StarletteHTTPException)
async def on_http_error(request: Request, exc: StarletteHTTPException):
    return error(exc.status_code, str(exc.detail))


@app.exception_handler(Exception)
async def on_unexpected(request: Request, exc: Exception):
    log.exception("unhandled error")
    return error(500, "internal error")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask(body: AskRequest, request: Request):
    try:
        answer = await run_in_threadpool(request.app.state.qa.ask, body.question)
    except GenerationError as e:
        log.error("generation failed: %s", e)
        return error(500, str(e))
    return answer.to_dict()
