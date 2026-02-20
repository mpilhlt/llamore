import logging
import os
import tempfile
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from llamore import (
    F1,
    GeminiExtractor,
    LineByLinePrompter,
    OpenaiExtractor,
    Reference,
    References,
    SchemaPrompter,
)
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

ALLOWED_API_KEY = os.getenv("ALLOWED_API_KEY")
if not ALLOWED_API_KEY:
    raise ValueError("ALLOWED_API_KEY environment variable must be set")


def api_error(detail: str, status_code: int = 400) -> HTTPException:
    """Create an HTTPException with logging."""
    logger.error(detail)
    return HTTPException(status_code=status_code, detail=detail)


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != ALLOWED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting llamore FastAPI application")
    yield
    logger.info("Shutting down llamore FastAPI application")


app = FastAPI(
    title="Llamore API",
    description="API for extracting and processing scholarly references using llamore",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc
    logger.error(
        f"Unhandled exception in {request.method} {request.url.path}:\n{traceback.format_exc()}"
    )
    return JSONResponse(
        status_code=500, content={"error": str(exc), "type": type(exc).__name__}
    )


# Request/Response Models
class ExtractTextRequest(BaseModel):
    text: str = Field(..., description="Text to extract references from")
    provider: str = Field("gemini", description="LLM provider: 'openai' or 'gemini'")
    provider_api_key: str = Field(..., description="API key for the LLM provider")
    model: Optional[str] = Field(None, description="Model name")
    prompter_type: str = Field("schema", description="'schema' or 'line_by_line'")
    step_by_step: bool = Field(
        False, description="Step-by-step extraction (schema only)"
    )
    additional_instructions: Optional[str] = Field(
        None, description="Additional instructions"
    )


class ReferencesResponse(BaseModel):
    references: List[Dict[str, Any]]
    count: int


class ToXMLRequest(BaseModel):
    references: List[Dict[str, Any]]
    pretty_print: bool = Field(True)


class XMLResponse(BaseModel):
    xml: str


class FromXMLRequest(BaseModel):
    xml: str


class F1Request(BaseModel):
    predictions: List[List[Dict[str, Any]]]
    labels: List[List[Dict[str, Any]]]
    levenshtein_distance: int | float = Field(0)
    metric_type: str = Field("macro", description="'macro' or 'micro'")


class F1Response(BaseModel):
    score: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None


# Utility functions
def create_extractor(
    provider: str,
    provider_api_key: str,
    model: Optional[str],
    prompter_type: str,
    step_by_step: bool = False,
):
    if not provider_api_key or not provider_api_key.strip():
        raise api_error(f"API key required for provider '{provider}'")

    prompter = (
        LineByLinePrompter()
        if prompter_type == "line_by_line"
        else SchemaPrompter(step_by_step=step_by_step)
    )

    if provider == "openai":
        return OpenaiExtractor(
            api_key=provider_api_key, model=model or "gpt-4o", prompter=prompter
        )
    elif provider == "gemini":
        return GeminiExtractor(
            api_key=provider_api_key,
            model=model or "gemini-2.5-flash",
            prompter=prompter,
        )
    else:
        raise api_error(f"Unsupported provider '{provider}'. Use 'openai' or 'gemini'")


def references_to_dict(references: References) -> List[Dict[str, Any]]:
    return [ref.model_dump(exclude_none=True) for ref in references]


def dict_to_references(refs_dict: List[Dict[str, Any]]) -> References:
    return References([Reference(**ref) for ref in refs_dict])


# Endpoints
@app.get("/")
async def root():
    return {
        "message": "Llamore API",
        "version": "1.0.0",
        "endpoints": {
            "extract_text": "/extract/text",
            "extract_pdf": "/extract/pdf",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "llamore-api"}


@app.post("/extract/text", response_model=ReferencesResponse)
async def extract_from_text(
    request: ExtractTextRequest, api_key: str = Depends(verify_api_key)
):
    if not request.text.strip():
        raise api_error("Text cannot be empty")

    try:
        extractor = create_extractor(
            request.provider,
            request.provider_api_key,
            request.model,
            request.prompter_type,
            request.step_by_step,
        )
        references = extractor(text=request.text)
    except HTTPException:
        raise
    except Exception as e:
        raise api_error(f"Extraction failed: {e}")

    logger.info(f"Extracted {len(references)} references from text")
    return ReferencesResponse(
        references=references_to_dict(references), count=len(references)
    )

@app.post("/extract/pdf", response_model=ReferencesResponse)
async def extract_from_pdf(
    file: UploadFile = File(...),
    provider: str = "gemini",
    provider_api_key: str = "",
    model: Optional[str] = None,
    prompter_type: str = "schema",
    step_by_step: bool = False,
    api_key: str = Depends(verify_api_key)
):
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise api_error("A valid .pdf file is required")

    content = await file.read()
    if not content:
        raise api_error("Uploaded file is empty")

    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix='.pdf', dir='/tmp') as tmp:
            tmp.write(content)
            tmp.flush()  # ensure bytes are written before extractor reads it
            tmp_path = Path(tmp.name)
            
            try:
                extractor = create_extractor(provider, provider_api_key, model, prompter_type, step_by_step)
                references = extractor(pdf=tmp_path)
            except HTTPException:
                raise
            except Exception as e:
                raise api_error(f"PDF extraction failed: {e}")

    except HTTPException:
        raise
    except Exception as e:
        raise api_error(f"PDF handling failed: {e}")

    logger.info(f"Extracted {len(references)} references from {file.filename}")
    return ReferencesResponse(references=references_to_dict(references), count=len(references))