# Multimodal RAG — API Guide for Frontend Developers

This guide explains how to talk to the Multimodal RAG backend so you can build a
chat UI, a search box, or a document viewer on top of it — without knowing
anything about the ML pipeline underneath.

The backend is a **FastAPI** service. It answers questions by searching a
vector database of PDFs (datasheets like the LM317 / LM2596) and generating an
answer with a multimodal LLM. Responses include the answer text **plus** the
supporting evidence: the exact document chunks, their tables, and their figures.

- Interactive docs: `GET /docs` (Swagger UI) — try every endpoint in the browser.
- Base URL: `http://localhost:8000` locally. In production, use the deployed URL
  (e.g. `https://your-app.up.railway.app`). Everything below is relative to it.

---

## 1. The endpoints at a glance

| Endpoint | Method | Purpose | Returns |
|---|---|---|---|
| `/` | GET | Index / endpoint list | `{name, version, endpoints}` |
| `/health` | GET | Is the service up? | `{status: "ok", vector_store: "loaded"}` |
| `/retrieve` | POST | Get evidence **without** an answer | `RetrieveResponse` |
| `/query` | POST | Get evidence **and** a full answer | `QueryResponse` |
| `/query/stream` | GET | Same as `/query` but token-by-token (SSE) | Server-Sent Events |
| `/images/{response_id}/{index}` | GET | Fetch one figure as an image | raw JPEG bytes |
| `/docs` | GET | Swagger UI (try-it-out docs) | HTML |

There are only **three shapes** you need to understand:

1. **Request** — always the same: `{"query": "...", "top_k": 10}`.
2. **`RetrieveResponse`** — the chunks + everything they contain.
3. **`QueryResponse`** — the same, plus `answer` and `answer_characters`.

---

## 2. The request

Both `/retrieve` and `/query` take the same JSON body:

```json
{
  "query": "Tell me about the pin configuration of the LM317.",
  "top_k": 10
}
```

| Field | Type | Rules | Meaning |
|---|---|---|---|
| `query` | string | 1–2000 chars | The question you want answered |
| `top_k` | integer | 1–30, default 10 | How many document chunks to fetch. More chunks = more evidence (and more images/tables), but slower. |

`top_k` is the only knob. Start with the default; raise it if answers miss
context, lower it if you need speed.

---

## 3. The response — what comes back

This is what `POST /query` returns (identical top-level shape for `/retrieve`,
minus the two `answer` fields):

```jsonc
{
  "query": "Tell me about the pin configuration of the LM317.",
  "num_chunks": 2,                 // how many chunks were retrieved
  "total_images": 3,               // sum of image_count across chunks
  "total_tables": 2,               // sum of tables across chunks
  "response_id": "79c107621a50474989f2c3eab59beb7a",  // key for image URLs
  "answer": "The LM317 is an adjustable linear voltage regulator...",  // /query only
  "answer_characters": 1086,       // /query only
  "chunks": [
    {
      "chunk_id": 1,               // 1-based position within this response
      "enhanced_content": "**SEARCHABLE DESCRIPTION: LM317 Voltage Regulator...", // markdown
      "raw_text": "Pin 1 (ADJ): adjustment pin...", // plain OCR text
      "tables_html": ["<table><tr><th>Pin</th><th>Name</th>...</table>"],
      "image_count": 2,
      "images_base64": ["/9j/4AAQSkZJRg==...", "/9j/4AAQSkZJRg==..."],
      "image_urls": ["/images/79c107621a50474989f2c3eab59beb7a/0",
                     "/images/79c107621a50474989f2c3eab59beb7a/1"],
      "has_table": true,
      "has_image": true,
      "source": "lm317_datasheet.pdf",
      "metadata": { }
    }
  ]
}
```

### What each chunk field means for your UI

| Field | Type | How to render it |
|---|---|---|
| `enhanced_content` | string | Markdown text. Render with a markdown parser (it uses `**bold**`, headings, lists). This is the search-friendly summary. |
| `raw_text` | string | Plain OCR text from the PDF. Use for a "raw source" toggle. |
| `tables_html` | string[] | Each entry is a full `<table>...</table>` HTML string. Inject as HTML (sanitise first) or parse into your own table component. |
| `images_base64` | string[] | Base64-encoded JPEGs (note the `data:image/jpeg` prefix is **not** included — add it yourself if needed). |
| `image_urls` | string[] | Relative HTTP paths to the same images. **Prefetch and absolutise these** (see §5). |
| `has_table` / `has_image` | boolean | Quick flags to skip rendering empty sections. |
| `source` | string \| null | Source filename, if the chunk knows it. |

### The two ways to get images (important)

Every figure exists in **two formats**:

1. **Inline base64** — `images_base64[i]`. It is a **base64-encoded JPEG string**,
   without the `data:` prefix. To display:
   ```js
   `<img src="data:image/jpeg;base64,${chunk.images_base64[i]}">`
   ```
   Pros: works forever, no extra request, no server state. Cons: big JSON payloads.

2. **HTTP URL** — `image_urls[i]`, e.g. `/images/79c10.../1`. Fetch it with
   `GET {base}/images/79c10.../1` and you get **raw JPEG bytes**
   (`Content-Type: image/jpeg`). Use it directly in `<img src>`:
   ```js
   `<img src="${base}${chunk.image_urls[i]}">`
   ```
   Pros: tiny JSON, browser-cacheable. Cons: **expires** (see §6).

Both are the **same JPEG**, decoded differently. Pick base64 when you must keep
the data around, URLs when you want clean, lightweight JSON.

---

## 4. Streaming the answer (`/query/stream`)

`GET /query/stream?query=...&top_k=10` is the same as `/query` but the answer
arrives as you type it. It's a standard **Server-Sent Events** stream
(`Content-Type: text/event-stream`), so in the browser you can use
`EventSource` directly — no library needed.

Because it's a GET, put the parameters in the **query string** and URL-encode:
`/query/stream?query=Tell%20me%20about%20the%20pin%20configuration%20of%20the%20LM317.`

Event flow, in order:

| Event | `data` payload | When |
|---|---|---|
| `retrieval` | a full `RetrieveResponse` (the chunks) | immediately, once |
| `token` | `{"delta": "The "}` | many times, one fragment of the answer each |
| `done` | `{"answer": "...", "answer_characters": 1086, "response_id": "...", "num_chunks": 2}` | after the last token |
| `error` | `{"message": "..."}` | only if something fails |

```js
const es = new EventSource(
  `${BASE}/query/stream?query=${encodeURIComponent("Tell me about the pin configuration of the LM317.")}&top_k=5`
);
es.addEventListener("retrieval", (e) => renderChunks(JSON.parse(e.data)));
es.addEventListener("token", (e) => appendToAnswer(JSON.parse(e.data).delta));
es.addEventListener("done", (e) => { es.close(); /* JSON.parse(e.data).answer is final */ });
es.addEventListener("error", (e) => console.error("stream failed", e));
```

Tip: build your answer by concatenating all `delta`s; the `done` event gives you
the authoritative final string.

---

## 5. Working example (plain JS)

```js
const BASE = "http://localhost:8000"; // or your deployed URL

async function ask(query, top_k = 10) {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function absolutize(url) {
  return url.startsWith("/") ? `${BASE}${url}` : url;
}

const data = await ask("Tell me about the pin configuration of the LM317.");
console.log(data.answer);                       // the generated answer
for (const chunk of data.chunks) {
  console.log(chunk.enhanced_content);          // markdown evidence
  console.log(chunk.tables_html);               // HTML table strings
  chunk.image_urls.forEach((u) => console.log(absolutize(u))); // absolute <img> URLs
}
```

**Always call `absolutize` on `image_urls`** — they are returned as relative
paths (`/images/...`) and must be prefixed with the base URL before a browser
can load them.

---

## 6. Dependencies & gotchas (read this)

- **`response_id` ties everything together.** Every chunk's `image_urls` is
  `/images/{response_id}/{index}`. The `index` is the figure's **global,
  0-based position across the whole response** — NOT per-chunk. Chunk 2's
  images start where chunk 1's ended. You never compute this yourself: each
  chunk's `image_urls` array is already the correct list for that chunk. Just
  take it as-is.

- **Image URLs expire.** The backend keeps figures in memory for ~30 minutes
  (`response_id` lives in a TTL cache). After that, `GET /images/{id}/{index}`
  returns `404 Not Found`. This is by design (memory management) — if you must
  keep images around longer, use `images_base64` instead, or re-run the query.

- **`response_id` is per-response.** Two different queries → two different ids,
  even for the same underlying document. Don't reuse an id from a previous
  response.

- **Missing / expired images → `404`.** Always handle image load errors in your
  UI (broken-image fallback), because the 30-minute TTL is real.

- **`raw_text` and `enhanced_content` are different.** `enhanced_content` is
  the AI-summarised, markdown version (nice to read). `raw_text` is the raw OCR
  (exact source wording, uglier). Prefer `enhanced_content` for display.

- **`tables_html` is raw HTML.** The strings come straight from the PDF
  parser. If you inject them with `innerHTML`, sanitise them first.

- **Streaming and the chunks arrive separately.** In `/query/stream`, the
  `retrieval` event contains the chunks; the answer tokens follow. If you want
  both in one place, combine the `retrieval` event's data with the `done`
  event's `answer`.

---

## 7. Prerequisites before you point a browser at this

- **CORS:** the API **allows browser access** from a configured allow-list of
  origins (currently `http://localhost:5173`, `http://localhost:3000`,
  `http://127.0.0.1:5173` — your local Vite dev server is covered). For a
  production frontend URL, the backend must add it to
  `CORS_ALLOWED_ORIGINS` (a comma-separated env var), e.g.
  `CORS_ALLOWED_ORIGINS=https://app.example.com,http://localhost:5173`.
  Non-browser clients are unaffected.

- **Health check:** always `GET /health` before hammering the API — the vector
  store loads lazily, so the very first request can take a few seconds.

- **Non-browser clients** (curl, Node, mobile apps) are unaffected by CORS.

---

## 8. Choosing an endpoint

| You want… | Use |
|---|---|
| A ready-to-display answer + evidence | `POST /query` |
| Evidence only (build your own answer elsewhere) | `POST /retrieve` |
| A typewriter/streaming answer effect | `GET /query/stream` |
| The figures of a response | `images_base64` (inline) or `image_urls` → `GET /images/...` |
| Just a health/probe check | `GET /health` |
