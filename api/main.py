"""Multimodal RAG HTTP API.

Exposes the retrieval stack over REST:

* ``POST /retrieve``       - chunks only (with tables + images).
* ``POST /query``          - full answer plus the supporting chunks.
* ``GET  /query/stream``   - SSE stream: retrieval summary, then answer tokens.
* ``GET  /images/{response_id}/{index}`` - serve a single figure as JPEG.

Run:
    uvicorn api.main:app --reload --port 8000

Interactive docs live at ``http://localhost:8000/docs``.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from api.schemas import QueryRequest, QueryResponse, RetrieveRequest, RetrieveResponse
from api.service import RAGService
from config.settings import CORS_ALLOWED_ORIGINS
from src.retrieval.generate import answer_query_stream

# ---------------------------------------------------------------------------
# Request-body example payloads (shown as a dropdown in /docs)
# ---------------------------------------------------------------------------
_REQUEST_BODY_EXAMPLES = [
    {
        "summary": "LM2596 pin configuration",
        "value": {"query": "Tell me about the pin configuration of the LM2596.", "top_k": 10},
    },
    {
        "summary": "LM317 pin configuration",
        "value": {"query": "Tell me about the pin configuration of the LM317.", "top_k": 5},
    },
]

_QUERY_PARAM_EXAMPLES = [
    "Tell me about the pin configuration of the LM2596.",
    "Tell me about the pin configuration of the LM317.",
]

_RICH_DESCRIPTION_COMMON = """
Ask a question about the indexed PDFs. The service:

1. embeds the question with the same model used at ingest time,
2. retrieves the `top_k` most relevant chunks from Chroma,
3. (for `/query` and `/query/stream`) asks the multimodal LLM to answer
   using the chunk text, tables and figures.

### Try it

```json
{
  "query": "Tell me about the pin configuration of the LM317.",
  "top_k": 10
}
```

### Response

Every response includes a `response_id`; chunk figures can be loaded over
HTTP via `GET /images/{response_id}/{index}`.
"""


def _request_body_examples(schema_name: str, examples: list[dict]) -> dict:
    """Build an ``openapi_extra`` requestBody with a proper Examples dropdown."""
    return {
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {"$ref": f"#/components/schemas/{schema_name}"},
                    "examples": {
                        f"example_{i + 1}": {
                            "summary": ex.get("summary", f"Example {i + 1}"),
                            "value": ex.get("value", ex),
                        }
                        for i, ex in enumerate(examples)
                    },
                }
            }
        }
    }


_REQUEST_BODY_EXAMPLES = [
    {
        "summary": "LM2596 pin configuration",
        "value": {"query": "Tell me about the pin configuration of the LM2596.", "top_k": 10},
    },
    {
        "summary": "LM317 pin configuration",
        "value": {"query": "Tell me about the pin configuration of the LM317.", "top_k": 5},
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Hold one RAG service (loaded lazily on first request)."""
    app.state.service = RAGService()
    yield


app = FastAPI(
    title="Multimodal RAG API",
    version="0.2.0",
    description=(
        "Retrieve chunks and generate multimodal answers from a PDF-based "
        "vector store.\n\n"
        "Every chunk carries its OCR text, extracted tables (HTML) and figures "
        "(inline base64 **and** fetchable image URLs). Try the example questions "
        "like *\"Tell me about the pin configuration of the LM317.\"*"
    ),
    lifespan=lifespan,
)

# Browser access: allow the configured origins (e.g. the local Vite dev server,
# and later the production frontend URL).  Non-browser callers are unaffected.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["meta"], summary="List endpoints")
def root() -> dict:
    """Human-readable index of what this API exposes."""
    return {
        "name": "Multimodal RAG API",
        "version": app.version,
        "docs": "/docs",
        "endpoints": {
            "POST /retrieve": "Retrieve chunks only (text, tables, images)",
            "POST /query": "Retrieve chunks and generate a full answer",
            "GET /query/stream": "SSE stream of the answer with retrieval summary",
            "GET /images/{response_id}/{index}": "Fetch a single figure as JPEG",
            "GET /health": "Service health / vector store status",
        },
    }


@app.get("/health", tags=["meta"], summary="Service health", response_description="Whether the vector store is loadable here")
async def health(request: Request) -> dict:
    """Report whether the vector store is loadable in this runtime.

    Useful as a startup probe: the first call may take a few seconds while the
    embedding model + Chroma store load into memory.
    """
    try:
        await request.app.state.service.ensure_ready()
        return {"status": "ok", "vector_store": "loaded"}
    except Exception as exc:
        return {"status": "error", "vector_store": "unavailable", "detail": str(exc)}


@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    tags=["retrieval"],
    summary="Retrieve chunks (no answer)",
    description=(
        "Embed the question, fetch the most relevant chunks and return them "
        "with their text, tables and images.\n\n"
        "Use this when you only need context (e.g. to feed another model)."
    )
    + _RICH_DESCRIPTION_COMMON,
    response_description="Retrieved chunks with multimodal content",
    openapi_extra=_request_body_examples("RetrieveRequest", _REQUEST_BODY_EXAMPLES),
)
async def retrieve(request: Request, req: RetrieveRequest) -> RetrieveResponse:
    """Retrieve chunks for a query; returns text, tables and images."""
    try:
        return await request.app.state.service.retrieve(req.query, req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["generation"],
    summary="Retrieve chunks and generate an answer",
    description=(
        "Everything /retrieve returns, plus a complete answer generated by "
        "the multimodal LLM from the retrieved chunks.\n\n"
        "```json\n"
        "{\n"
        "  \"query\": \"Tell me about the pin configuration of the LM317.\",\n"
        "  \"top_k\": 10\n"
        "}\n"
        "```\n\n"
        "Answers are grounded in the retrieved chunks; no web search is used."
    )
    + _RICH_DESCRIPTION_COMMON,
    response_description="Retrieved chunks plus the generated answer",
    openapi_extra=_request_body_examples("QueryRequest", _REQUEST_BODY_EXAMPLES),
)
async def query(request: Request, req: QueryRequest) -> QueryResponse:
    """Retrieve chunks and generate a full answer from the multimodal LLM."""
    try:
        answer, base = await request.app.state.service.answer(req.query, req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return QueryResponse(**base.model_dump(), answer=answer, answer_characters=len(answer))


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Events frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get(
    "/query/stream",
    tags=["generation"],
    summary="Stream an answer over SSE",
    description=(
        "Streams the answer token-by-token over Server-Sent Events. Event flow:\n\n"
        "1. ``event: retrieval`` - full chunk payloads (like /retrieve),\n"
        "2. ``event: token``    - one per answer fragment (field ``delta``),\n"
        "3. ``event: done``     - final answer + ``response_id``,\n"
        "4. ``event: error``    - only on failure.\n\n"
        "Open the URL directly in a browser to watch the stream, or consume it "
        "with any SSE client. Works with e.g. ``curl -N``."
    ),
    response_description="text/event-stream of retrieval + answer tokens",
)
async def query_stream(
    request: Request,
    query: str = Query(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question about the documents.",
        examples=_QUERY_PARAM_EXAMPLES,
    ),
    top_k: int = Query(
        10,
        ge=1,
        le=30,
        description="How many chunks to retrieve from the vector store.",
        examples=[10],
    ),
) -> StreamingResponse:
    """Stream an answer over SSE.

    Emits a ``retrieval`` event (full chunk payloads), then one ``token`` event
    per answer chunk, then a final ``done`` event.  On failure an ``error``
    event is emitted instead.
    """
    service = request.app.state.service
    await service.ensure_ready()
    retriever = service.get_retriever(top_k)

    def generate():
        try:
            stream = answer_query_stream(retriever, query)
            chunks = stream.chunks
            base = service.to_retrieve_response(query, chunks)
            yield _sse("retrieval", base.model_dump())

            collected: list[str] = []
            for token in stream:
                collected.append(token)
                yield _sse("token", {"delta": token})

            answer = "".join(collected)
            yield _sse(
                "done",
                {
                    "answer": answer,
                    "answer_characters": len(answer),
                    "response_id": base.response_id,
                    "num_chunks": base.num_chunks,
                },
            )
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get(
    "/images/{response_id}/{index}",
    tags=["images"],
    summary="Fetch a figure as JPEG",
    description=(
        "Serves one figure referenced by a chunk's ``image_urls``.\n\n"
        "- ``response_id`` comes from the retrieve/query response and groups "
        "all figures of that response.\n"
        "- ``index`` is the **global, 0-based** position of the image across "
        "the whole response (chunk 2 of a 2-image chunk 1 starts at index 2).\n\n"
        "```\n"
        "GET /images/79c107621a50474989f2c3eab59beb7a/1\n"
        "```\n"
        "Returns the figure as ``image/jpeg``. Entries are cached for 10 "
        "minutes and expire after ~30; expired or unknown ids return 404."
    ),
    response_description="image/jpeg bytes of the requested figure",
    responses={404: {"description": "Unknown response_id or expired / out-of-range index"}},
)
async def serve_image(response_id: str, index: int, request: Request) -> Response:
    """Serve one figure from a previous retrieve/query response.

    Entries expire after ~30 minutes; expired or unknown ids return 404.
    """
    raw = request.app.state.service.images.get(response_id, index)
    if raw is None:
        raise HTTPException(status_code=404, detail="Image not found or expired")
    return Response(
        content=raw,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=600"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
