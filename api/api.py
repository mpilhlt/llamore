import logging
import os
import secrets
import tempfile
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from llamore import (
    GeminiExtractor,
    LineByLinePrompter,
    OpenaiExtractor,
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

# ===== Config =====

ALLOWED_API_KEY = os.getenv("ALLOWED_API_KEY")
if not ALLOWED_API_KEY:
    raise ValueError("ALLOWED_API_KEY environment variable must be set")

MAX_PDF_SIZE_BYTES = int(os.getenv("MAX_PDF_SIZE_MB", "50")) * 1024 * 1024


# ===== Auth =====

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
provider_key_header = APIKeyHeader(name="X-Provider-Key", auto_error=False)


def api_error(detail: str, status_code: int = 400) -> HTTPException:
    """Create an HTTPException with server-side logging."""
    logger.error(detail)
    return HTTPException(status_code=status_code, detail=detail)


async def verify_api_key(api_key: str = Security(api_key_header)):
    # secrets.compare_digest prevents timing-based attacks
    if not api_key or not secrets.compare_digest(api_key, ALLOWED_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


async def verify_provider_key(provider_api_key: str = Security(provider_key_header)):
    if not provider_api_key or not provider_api_key.strip():
        raise HTTPException(status_code=401, detail="Missing or empty provider API key")
    return provider_api_key


# ===== App =====


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
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )


# ===== Schemas =====


class BaseExtractionConfig(BaseModel):
    """Options shared across all providers and input types."""

    prompter_type: Literal["schema", "line_by_line"] = Field(
        "schema",
        description="Prompter type for extraction.",
    )
    step_by_step: bool = Field(
        False,
        description="Enable step-by-step extraction (SchemaPrompter only).",
    )
    extra_api_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra keyword arguments forwarded to the provider's generate call.",
    )
    return_xml: bool = Field(
        False,
        description="If true, also return a TEI XML representation of the extracted references.",
    )


class OpenaiExtractionConfig(BaseExtractionConfig):
    """OpenAI-specific extraction options."""

    model: str = Field(
        "gpt-4o",
        description="OpenAI model name.",
    )
    endpoint: Literal["create", "parse"] = Field(
        "create",
        description=(
            "'parse' uses beta.chat.completions.parse for native structured output "
            "and requires a compatible model. "
            "Cannot be combined with prompter_type='line_by_line'."
        ),
    )
    client_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra keyword arguments forwarded to the openai.OpenAI() constructor "
            "(e.g. base_url for Ollama/vLLM/SGLang-compatible endpoints, "
            "timeout, max_retries, default_headers)."
        ),
    )


class GeminiExtractionConfig(BaseExtractionConfig):
    """Gemini-specific extraction options."""

    model: str = Field(
        "gemini-2.5-flash",
        description="Gemini model name.",
    )


class OpenaiExtractTextRequest(OpenaiExtractionConfig):
    """Request body for OpenAI text extraction."""

    text: str = Field(
        ...,
        min_length=1,
        description="Raw text from which references should be extracted.",
    )
    additional_instructions: Optional[str] = Field(
        None,
        description=(
            "Additional instructions appended to the user prompt. "
            "Only effective when using the 'line_by_line' prompter; "
            "ignored by the 'schema' prompter."
        ),
    )


class GeminiExtractTextRequest(GeminiExtractionConfig):
    """Request body for Gemini text extraction."""

    text: str = Field(
        ...,
        min_length=1,
        description="Raw text from which references should be extracted.",
    )
    additional_instructions: Optional[str] = Field(
        None,
        description=(
            "Additional instructions appended to the user prompt. "
            "Only effective when using the 'line_by_line' prompter; "
            "ignored by the 'schema' prompter."
        ),
    )


class ReferencesResponse(BaseModel):
    """Response containing extracted references and optional TEI XML."""

    references: List[Dict[str, Any]] = Field(
        ...,
        description="List of extracted references as JSON objects.",
    )
    xml: Optional[str] = Field(
        None,
        description="TEI XML representation of references (only present if return_xml=True).",
    )


# ===== Factories =====


def _build_prompter(
    prompter_type: Literal["schema", "line_by_line"],
    step_by_step: bool,
    endpoint: Literal["create", "parse"] = "create",
):
    if prompter_type == "line_by_line":
        if endpoint == "parse":
            raise api_error(
                "The 'parse' endpoint is incompatible with the 'line_by_line' prompter."
            )
        return LineByLinePrompter()
    elif prompter_type == "schema":
        return SchemaPrompter(step_by_step=step_by_step)
    else:
        raise api_error(
            f"Unsupported prompter_type '{prompter_type}'. Choose 'schema' or 'line_by_line'."
        )


def create_openai_extractor(
    provider_api_key: str,
    config: OpenaiExtractionConfig,
) -> OpenaiExtractor:
    prompter = _build_prompter(
        config.prompter_type, config.step_by_step, config.endpoint
    )
    return OpenaiExtractor(
        api_key=provider_api_key,
        model=config.model,
        prompter=prompter,
        endpoint=config.endpoint,
        **config.client_kwargs,
    )


def create_gemini_extractor(
    provider_api_key: str,
    config: GeminiExtractionConfig,
) -> GeminiExtractor:
    prompter = _build_prompter(config.prompter_type, config.step_by_step)
    return GeminiExtractor(
        api_key=provider_api_key,
        model=config.model,
        prompter=prompter,
    )


def references_to_response(
    references: References, return_xml: bool
) -> ReferencesResponse:
    refs_dict = [ref.model_dump(exclude_none=True) for ref in references]
    xml: Optional[str] = None
    if return_xml and references:
        try:
            xml = references.to_xml(pretty_print=True)
        except Exception:
            logger.warning("Failed to convert references to TEI XML.", exc_info=True)
    return ReferencesResponse(references=refs_dict, xml=xml)


async def _read_and_validate_pdf(file: UploadFile) -> bytes:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise api_error("A valid .pdf file is required.")
    content = await file.read()
    if not content:
        raise api_error("Uploaded file is empty.")
    if len(content) > MAX_PDF_SIZE_BYTES:
        raise api_error(
            f"PDF exceeds the maximum allowed size of {MAX_PDF_SIZE_BYTES // (1024 * 1024)} MB.",
            status_code=413,
        )
    return content


async def _run_pdf_extraction(
    extractor,
    file: UploadFile,
    extra_api_kwargs: Dict[str, Any],
    return_xml: bool,
) -> ReferencesResponse:
    content = await _read_and_validate_pdf(file)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            references = extractor(pdf=tmp_path, **extra_api_kwargs)
        except HTTPException:
            raise
        except Exception:
            logger.error(
                "PDF extraction failed for '%s'.", file.filename, exc_info=True
            )
            raise api_error(
                "Reference extraction failed. Check server logs for details."
            )
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                logger.warning(
                    "Could not delete temporary file '%s'.", tmp_path, exc_info=True
                )

    logger.info("Extracted %d references from '%s'.", len(references), file.filename)
    return references_to_response(references, return_xml)


# ===== Endpoints =====


@app.get("/")
async def root():
    return {
        "message": "Llamore API",
        "version": "1.0.0",
        "endpoints": {
            "extract_openai_text": "/extract/openai/text",
            "extract_openai_pdf": "/extract/openai/pdf",
            "extract_gemini_text": "/extract/gemini/text",
            "extract_gemini_pdf": "/extract/gemini/pdf",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "llamore-api"}


@app.post("/extract/openai/text", response_model=ReferencesResponse)
async def extract_openai_text(
    request: OpenaiExtractTextRequest,
    provider_api_key: str = Depends(verify_provider_key),
    _: str = Depends(verify_api_key),
):
    """Extract references from plain text using OpenAI."""
    if not request.text.strip():
        raise api_error("Text cannot be empty.")
    try:
        extractor = create_openai_extractor(provider_api_key, request)
        references = extractor(
            text=request.text,
            additional_instructions=request.additional_instructions,
            **request.extra_api_kwargs,
        )
    except HTTPException:
        raise
    except Exception:
        logger.error("Text extraction failed.", exc_info=True)
        raise api_error("Reference extraction failed. Check server logs for details.")

    logger.info("Extracted %d references from text.", len(references))
    return references_to_response(references, request.return_xml)


@app.post("/extract/openai/pdf", response_model=ReferencesResponse)
async def extract_openai_pdf(
    file: UploadFile,
    request: OpenaiExtractionConfig = Depends(),
    provider_api_key: str = Depends(verify_provider_key),
    _: str = Depends(verify_api_key),
):
    """Extract references from a PDF file using OpenAI."""
    try:
        extractor = create_openai_extractor(provider_api_key, request)
    except HTTPException:
        raise
    return await _run_pdf_extraction(
        extractor, file, request.extra_api_kwargs, request.return_xml
    )


@app.post("/extract/gemini/text", response_model=ReferencesResponse)
async def extract_gemini_text(
    request: GeminiExtractTextRequest,
    provider_api_key: str = Depends(verify_provider_key),
    _: str = Depends(verify_api_key),
):
    """Extract references from plain text using Gemini."""
    if not request.text.strip():
        raise api_error("Text cannot be empty.")
    try:
        extractor = create_gemini_extractor(provider_api_key, request)
        references = extractor(
            text=request.text,
            additional_instructions=request.additional_instructions,
            **request.extra_api_kwargs,
        )
    except HTTPException:
        raise
    except Exception:
        logger.error("Text extraction failed.", exc_info=True)
        raise api_error("Reference extraction failed. Check server logs for details.")

    logger.info("Extracted %d references from text.", len(references))
    return references_to_response(references, request.return_xml)


@app.post("/extract/gemini/pdf", response_model=ReferencesResponse)
async def extract_gemini_pdf(
    file: UploadFile,
    request: GeminiExtractionConfig = Depends(),
    provider_api_key: str = Depends(verify_provider_key),
    _: str = Depends(verify_api_key),
):
    """Extract references from a PDF file using Gemini."""
    try:
        extractor = create_gemini_extractor(provider_api_key, request)
    except HTTPException:
        raise
    return await _run_pdf_extraction(
        extractor, file, request.extra_api_kwargs, request.return_xml
    )
