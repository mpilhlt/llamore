import logging
import os
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

ALLOWED_API_KEY = os.getenv("ALLOWED_API_KEY")
if not ALLOWED_API_KEY:
    raise ValueError("ALLOWED_API_KEY environment variable must be set")


def api_error(detail: str, status_code: int = 400) -> HTTPException:
    """Create an HTTPException with logging."""
    logger.error(detail)
    return HTTPException(status_code=status_code, detail=detail)


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
provider_key_header = APIKeyHeader(name="X-Provider-Key", auto_error=False)


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


# ===== Base Request and Response Classes =====


class ExtractionConfig(BaseModel):
    """Common extraction configuration options."""

    provider: Literal["openai", "gemini"] = Field(
        "gemini", description="LLM provider to use for extraction"
    )
    model: Optional[str] = Field(
        None,
        description="Model name. Defaults to gemini-2.5-flash for Gemini, gpt-4o for OpenAI",
    )
    prompter_type: Literal["schema", "line_by_line"] = Field(
        "schema", description="Prompter type for extraction"
    )
    step_by_step: bool = Field(
        False, description="Enable step-by-step extraction (only for schema prompter)"
    )

    additional_instructions: Optional[str] = Field(
        None, description="Additional instructions for the extractor"
    )

    return_xml: bool = Field(
        False, description="Convert extracted references to TEI XML format in response"
    )


# ===== Response Classes =====


class ReferencesResponse(BaseModel):
    """Response containing extracted references and optional XML."""

    references: List[Dict[str, Any]] = Field(
        ..., description="List of extracted references"
    )
    xml: Optional[str] = Field(
        None,
        description="TEI XML representation of references (only if return_xml=True)",
    )


# Utility functions
def create_extractor(
    provider: Literal["openai", "gemini"],
    provider_api_key: str,
    model: Optional[str],
    prompter_type: Literal["schema", "line_by_line"],
    step_by_step: bool = False,
):
    """Create an extractor instance based on provider and configuration.

    Args:
        provider: The LLM provider (openai or gemini)
        provider_api_key: API key for the provider
        model: Model name (optional, uses defaults if not specified)
        prompter_type: Type of prompter to use (schema or line_by_line)
        step_by_step: Enable step-by-step extraction (schema only)

    Returns:
        An extractor instance (OpenaiExtractor or GeminiExtractor)

    Raises:
        HTTPException: If provider is unsupported or API key is invalid
    """

    if not provider_api_key or not provider_api_key.strip():
        raise api_error(f"API key required for provider '{provider}'")

    if prompter_type == "line_by_line":
        prompter = LineByLinePrompter()
    elif prompter_type == "schema":
        prompter = SchemaPrompter(step_by_step=step_by_step)
    else:
        raise api_error(
            f"Unsupported prompter type '{prompter_type}'. Use 'schema' or 'line_by_line'"
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


class ExtractTextRequest(ExtractionConfig):
    """Request to extract references from plain text."""

    text: str = Field(..., min_length=1, description="Text to extract references from")


@app.post("/extract/text", response_model=ReferencesResponse)
async def extract_from_text(
    request: ExtractTextRequest,
    provider_api_key: str = Security(provider_key_header),
    api_key: str = Depends(verify_api_key),
):
    if not request.text.strip():
        raise api_error("Text cannot be empty")

    try:
        extractor = create_extractor(
            request.provider,
            provider_api_key,
            request.model,
            request.prompter_type,
            request.step_by_step,
        )
        references = extractor(
            text=request.text, additional_instructions=request.additional_instructions
        )
    except HTTPException:
        raise
    except Exception as e:
        raise api_error(f"Extraction failed: {e}")

    logger.info(f"Extracted {len(references)} references from text")

    response_data = {"references": references_to_dict(references), "xml": None}

    if request.return_xml and references:
        try:
            response_data["xml"] = references.to_xml(pretty_print=True)
        except Exception as e:
            # Don't fail the whole request, just omit the XML
            logger.warning(f"Failed to convert references to XML: {e}")

    return ReferencesResponse(**response_data)


class ExtractPdfRequest(ExtractionConfig):
    """Request to extract references from a PDF file.

    Note: The file itself is passed as form data via UploadFile.
    """

    file: UploadFile = Field(..., description="PDF file to extract references from")


@app.post("/extract/pdf", response_model=ReferencesResponse)
async def extract_from_pdf(
    request: ExtractPdfRequest = Depends(),
    provider_api_key: str = Security(provider_key_header),
    api_key: str = Depends(verify_api_key),
):
    """Extract references from a PDF file.

    Args:
        request: ExtractPdfRequest containing file and extraction parameters
        provider_api_key: LLM provider API key from X-Provider-Key header
        api_key: Verified API key from X-API-Key header

    Returns:
        ReferencesResponse with extracted references and optional XML
    """
    file = request.file
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise api_error("A valid .pdf file is required")

    content = await file.read()
    if not content:
        raise api_error("Uploaded file is empty")

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf", dir="/tmp"
        ) as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_path = Path(tmp.name)

            try:
                extractor = create_extractor(
                    request.provider,
                    provider_api_key,
                    request.model,
                    request.prompter_type,
                    request.step_by_step,
                )
                references = extractor(pdf=tmp_path)
            except HTTPException:
                raise
            except Exception as e:
                raise api_error(f"PDF extraction failed: {e}")
            finally:
                # Clean up temp file
                try:
                    tmp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to clean up temporary file: {cleanup_error}"
                    )

    except HTTPException:
        raise
    except Exception as e:
        raise api_error(f"PDF handling failed: {e}")

    logger.info(f"Extracted {len(references)} references from {file.filename}")

    response_data = {"references": references_to_dict(references), "xml": None}

    if request.return_xml and references:
        try:
            response_data["xml"] = references.to_xml(pretty_print=True)
        except Exception as e:
            logger.warning(f"Failed to convert references to XML: {e}")
            # Don't fail the whole request, just omit the XML

    return ReferencesResponse(**response_data)
