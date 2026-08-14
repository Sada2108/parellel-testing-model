"""Pydantic request/response models for the FastAPI layer.

Every model carries rich descriptions and real-world examples so the OpenAPI
docs (``/docs``) are self-explanatory and instantly try-able.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared request examples (used across all query-style endpoints)
# ---------------------------------------------------------------------------
QUERY_EXAMPLES: list[dict] = [
    {
        "query": "Tell me about the pin configuration of the LM2596.",
        "top_k": 10,
    },
    {
        "query": "Tell me about the pin configuration of the LM317.",
        "top_k": 5,
    },
]


class QueryRequest(BaseModel):
    """A retrieval/generation request.

    ``top_k`` overrides the default retrieval size for this call without
    rebuilding the vector store.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description=(
            "The user's question about the documents. The model retrieves the "
            "most relevant chunks and answers using their text, tables and images."
        ),
        examples=[
            "Tell me about the pin configuration of the LM317.",
            "Tell me about the pin configuration of the LM2596.",
        ],
    )
    top_k: int = Field(
        10,
        ge=1,
        le=30,
        description="How many chunks to retrieve from the vector store.",
        examples=[10],
    )

    model_config = ConfigDict(json_schema_extra={"examples": QUERY_EXAMPLES})


class RetrieveRequest(QueryRequest):
    """Same shape as :class:`QueryRequest`; used by the retrieve-only endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"query": "Tell me about the pin configuration of the LM2596.", "top_k": 10}
            ]
        }
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class ChunkPayload(BaseModel):
    """One retrieved chunk with its full multimodal content.

    Images are returned twice so callers can pick: inline base64 for quick
    display, or ``image_urls`` to load each figure over HTTP (the URLs point
    at ``GET /images/{response_id}/{index}``). The ``index`` is the image's
    global position across the whole response (0-based), so a chunk's
    ``image_urls`` list is always the correct list for that chunk.
    """

    chunk_id: int = Field(..., description="1-based position in this response")
    enhanced_content: str = Field(
        ..., description="AI-enhanced searchable summary of the chunk"
    )
    raw_text: str = Field("", description="Original OCR text extracted from the PDF")
    tables_html: list[str] = Field(default_factory=list, description="HTML table markup")
    image_count: int = Field(0, description="Number of images in this chunk")
    images_base64: list[str] = Field(default_factory=list, description="Inline base64 images")
    image_urls: list[str] = Field(
        default_factory=list,
        description="HTTP URLs to fetch each image (see /images/{response_id}/{index})",
    )
    has_table: bool = False
    has_image: bool = False
    source: str | None = Field(None, description="Source document, if known")
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_id": 1,
                "enhanced_content": (
                    "The LM317 is an adjustable linear voltage regulator. "
                    "Its three pins are: ADJ (adjust), VOUT (output) and "
                    "VIN (input). The output voltage is set by the resistor "
                    "divider on the ADJ pin."
                ),
                "raw_text": (
                    "Pin 1 (ADJ): adjustment pin used to set output voltage via "
                    "a resistor divider. Pin 2 (VOUT): regulated output. "
                    "Pin 3 (VIN): unregulated input."
                ),
                "tables_html": [
                    "<table><tr><th>Pin</th><th>Name</th><th>Function</th></tr>"
                    "<tr><td>1</td><td>ADJ</td><td>Adjust</td></tr></table>"
                ],
                "image_count": 2,
                "images_base64": ["<base64-encoded-jpeg>", "<base64-encoded-jpeg>"],
                "image_urls": [
                    "/images/79c107621a50474989f2c3eab59beb7a/0",
                    "/images/79c107621a50474989f2c3eab59beb7a/1",
                ],
                "has_table": True,
                "has_image": True,
                "source": "lm317_datasheet.pdf",
                "metadata": {"has_table": True, "has_image": True},
            }
        }
    )


class RetrieveResponse(BaseModel):
    """Result of a retrieve-only request (no answer generation)."""

    query: str
    num_chunks: int = Field(..., description="Number of chunks returned")
    total_images: int = Field(..., description="Images across all chunks")
    total_tables: int = Field(..., description="Tables across all chunks")
    response_id: str = Field(
        ...,
        description=(
            "Key for the image URLs. Every image in this response lives at "
            "/images/{response_id}/{index} with a 0-based global index."
        ),
    )
    chunks: list[ChunkPayload] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Tell me about the pin configuration of the LM317.",
                "num_chunks": 2,
                "total_images": 2,
                "total_tables": 1,
                "response_id": "79c107621a50474989f2c3eab59beb7a",
                "chunks": [
                    {
                        "chunk_id": 1,
                        "enhanced_content": "The LM317 is an adjustable linear voltage regulator...",
                        "raw_text": "Pin 1 (ADJ): ...",
                        "tables_html": ["<table>..."],
                        "image_count": 2,
                        "images_base64": ["<base64-encoded-jpeg>", "<base64-encoded-jpeg>"],
                        "image_urls": [
                            "/images/79c107621a50474989f2c3eab59beb7a/0",
                            "/images/79c107621a50474989f2c3eab59beb7a/1",
                        ],
                        "has_table": True,
                        "has_image": True,
                        "source": "lm317_datasheet.pdf",
                        "metadata": {"has_table": True, "has_image": True},
                    }
                ],
            }
        }
    )


class QueryResponse(RetrieveResponse):
    """Result of a full query: retrieved chunks plus the generated answer."""

    answer: str = Field(..., description="The LLM-generated answer")
    answer_characters: int = Field(..., description="Length of the answer in characters")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Tell me about the pin configuration of the LM317.",
                "num_chunks": 2,
                "total_images": 2,
                "total_tables": 1,
                "response_id": "79c107621a50474989f2c3eab59beb7a",
                "answer": (
                    "The LM317 is an adjustable linear voltage regulator with "
                    "three pins. Pin 1 is the adjust (ADJ) pin, used to set the "
                    "output voltage with a resistor divider. Pin 2 is the "
                    "regulated output (VOUT), and pin 3 is the unregulated input "
                    "(VIN). The output voltage is approximately 1.25 V times "
                    "(1 + R2/R1)."
                ),
                "answer_characters": 340,
                "chunks": [],
            }
        }
    )
