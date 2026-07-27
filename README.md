<div align="center">

# 🧠 ragent

### *A guardrail-first policy assistant that knows when to give up and call a human* 🙋‍♂️

*Multi-turn LangGraph agent · hybrid RAG · frustration-aware escalation · async ingestion pipeline*

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-state%20machine-1C3C3C?logo=langchain&logoColor=white)
![Claude](https://img.shields.io/badge/LLM-Claude-D97757?logo=anthropic&logoColor=white)
![Chroma](https://img.shields.io/badge/VectorDB-Chroma-FF6F00)
![Celery](https://img.shields.io/badge/Queue-Celery%20%2B%20Redis-37814A?logo=celery&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

</div>

---

## 👋 What even is this thing?

`ragent` answers "what does the policy say about X?" questions over your own PDFs/DOCX policy docs — but it's built around two questions most weekend-project RAG bots don't bother asking:

1. 🧯 **"Is the model actually making this up?"** → input/output guardrails + a lexical grounding check + an LLM judge + RAGAS, so hallucinations get caught instead of shipped.
2. 😤 **"Is the user about to lose their mind?"** → a cumulative, regex-scored frustration meter that auto-escalates to a human the moment things cross a threshold — no vibes-based LLM judgment call required.

It's exposed three ways off the *same* agent graph:

| Surface | Use it for |
|---|---|
| 🌐 **FastAPI** (`/api/v1/ask`, `/ask/stream`, `/ingest`) | Talking to it over HTTP, streaming tokens via SSE |
| 🔌 **MCP server** (`src/mcp_server.py`) | Plugging policy search into Claude Desktop or any MCP client |
| 🐍 **Direct Python** (`ask()`, `ask_with_session()`) | Embedding the agent straight into another service |

Built as a personal deep-dive into LangGraph state machines, hybrid retrieval, and "what does production-grade actually require" — so expect a few 🚧 rough edges alongside the parts that are genuinely solid.

---

## 🗺️ Table of Contents

- [🏗️ Architecture](#️-architecture)
- [🔁 Diagram Zoo (learning reference)](#-diagram-zoo-learning-reference)
  - [Request lifecycle — sync `/ask`](#1-request-lifecycle--sync-ask)
  - [Async ingestion via Celery + Redis](#2-async-ingestion-via-celery--redis)
  - [Frustration score state machine](#3-frustration-score-state-machine)
  - [Middleware pipeline](#4-middleware-pipeline)
- [🧰 Tech Stack & Why](#-tech-stack--why)
- [💡 Core Design Decisions](#-core-design-decisions)
- [📁 Module Map](#-module-map)
- [⚙️ Configuration Reference](#️-configuration-reference)

---

## 🏗️ Architecture

```mermaid
flowchart TB
    subgraph Clients["🚪 Entry Points"]
        A1["FastAPI\n/ask, /ask/stream"]
        A2["MCP Server (stdio)\nsearch_policies / list_policy_documents"]
        A3["Direct Python API\nask() / ask_with_session()"]
    end

    subgraph MW["🛡️ HTTP Middleware Stack"]
        RID["RequestID"] --> RL["RateLimit\n(sliding window / IP)"] --> CORS["CORS"] --> LOG["Structured JSON logging"]
    end

    subgraph Core["🧠 Master Agent — LangGraph StateGraph"]
        direction TB
        IG["input_guard\nPII + prompt-injection filter"]
        FC["frustration_check\nregex signal scan + score update"]
        RT["router\nLLM structured-output classifier"]
        PA["policy_agent (sub-graph)"]
        OOS["out_of_scope_handler"]
        OG["output_guard\nPII redaction + grounding check"]
        HE["human_escalation 🆘\nticket stub"]
    end

    subgraph SubGraph["📚 policy_agent sub-graph"]
        direction TB
        TQ["transform_query"] --> RD["retrieve"] --> GN["generate\n(ChatAnthropic)"]
    end

    subgraph RAGLayer["🔍 RAG Layer"]
        VS[("Chroma\npersistent + singleton")]
        EMB["Embeddings\nHF BGE (local) or OpenAI"]
        BM["BM25Retriever\nkeyword search"]
        ENS["EnsembleRetriever\n0.6 semantic / 0.4 BM25"]
    end

    subgraph AsyncPipe["⚙️ Async Ingestion Pipeline"]
        REDIS[("Redis\nbroker + result backend")]
        WORKER["Celery worker\nload → chunk → embed → store"]
    end

    A1 --> MW --> IG
    A3 --> IG
    A2 -.direct retriever call, skips guardrails.-> ENS

    IG -->|blocked| END1(["END"])
    IG --> FC
    FC -->|score ≥ 0.80| HE
    FC -->|below threshold| RT
    RT -->|intent=policy| PA
    RT -->|intent=out_of_scope| OOS
    PA --> OG --> END2(["END"])
    OOS --> END2
    HE --> END2

    PA -.contains.-> SubGraph
    RD --> ENS --> VS
    ENS --> BM
    VS --> EMB

    A1 -."POST /ingest".-> REDIS --> WORKER --> VS

    CKPT[("SqliteSaver\n(fallback: MemorySaver)")]
    Core -. persists frustration_score,\nescalation_requested, per session_id .- CKPT
```


---

## 🔁 Design Zoo

A handful of extra diagrams purely so future-me (or anyone poking around this repo) can see *how* each moving part behaves, not just that it exists.

### 1. Request lifecycle — sync `/ask`

```mermaid
sequenceDiagram
    participant C as Client
    participant MW as Middleware
    participant API as FastAPI route
    participant SEM as asyncio.Semaphore
    participant G as master_agent (LangGraph)
    participant IG as input_guard
    participant FR as frustration_check
    participant RT as router (LLM)
    participant PA as policy_agent
    participant VS as Chroma
    participant LLM as ChatAnthropic
    participant OG as output_guard

    C->>MW: POST /ask {query, session_id?}
    MW->>MW: assign request_id, check rate limit
    MW->>API: forward request
    API->>SEM: acquire (max_concurrent_llm_calls)
    API->>G: ainvoke(turn_input, thread_id=session_id)\nwrapped in asyncio.wait_for(timeout)
    G->>IG: check_input(state)
    alt PII or injection detected
        IG-->>G: error + canned final_response
        G-->>API: return blocked response
    else clean
        IG-->>G: sanitized_query
        G->>FR: detect_frustration_signals + update score
        alt score ≥ 0.80
            FR-->>G: escalation_requested
            G->>G: human_escalation_node 🆘
            G-->>API: ticket-reference message
        else
            FR-->>G: continue
            G->>RT: classify(sanitized_query)
            RT-->>G: intent (policy | out_of_scope)
            alt intent == policy
                G->>PA: run sub-graph
                PA->>VS: retriever.invoke(query)
                VS-->>PA: top-k Document chunks
                PA->>LLM: generate(context + question)
                LLM-->>PA: generated_response
                PA-->>G: state update
                G->>OG: check_output(state)
                OG-->>G: final_response + guardrail_flags
            else out_of_scope
                G->>G: out_of_scope_handler_node
            end
            G-->>API: AskResponse
        end
    end
    API->>SEM: release
    API-->>C: {response, guardrail_flags, sources?, session_id}
```

The **streaming** twin (`/ask/stream`) runs the identical graph via `astream_events(..., version="v2")`, filters for `on_chat_model_stream` events scoped to the `generate` node only (so router/judge tokens never leak to the client), and emits SSE with a final `done: true` frame.

### 2. Async ingestion via Celery + Redis

Swapped `BackgroundTasks` for a real queue. 🎉

```mermaid
sequenceDiagram
    participant U as You
    participant API as POST /ingest
    participant R as Redis (broker)
    participant W as Celery worker
    participant VS as Chroma

    U->>API: {"directory": "data/policies", "clear": false}
    API->>R: ingest_documents.delay(dir, clear)
    R-->>API: task_id
    API-->>U: 202 Accepted {task_id, status: "queued"}

    Note over W,R: worker polls Redis, picks up the task

    R->>W: dequeue ingest_documents
    W->>W: update_state(step="clearing", pct=0)
    opt clear=true
        W->>VS: delete_collection()
    end
    W->>W: update_state(step="loading", pct=20)
    W->>W: load_directory() → chunk with RecursiveCharacterTextSplitter
    W->>W: update_state(step="embedding", pct=50)
    W->>VS: add_documents(chunks)
    W->>W: update_state(step="done", pct=100)
    W-->>R: result {chunks_written, cleared}

    U->>API: GET /ingest/status/{task_id}
    API->>R: AsyncResult(task_id).state / .info
    R-->>API: {state, pct, step, chunks_written}
    API-->>U: IngestStatusResponse
```

**Why this over `BackgroundTasks`:** crash-safe redelivery (`task_acks_late=True`), automatic retry with exponential backoff on transient I/O errors, real progress percentages instead of "check the logs," and the ability to add more worker processes without touching the API process at all.

### 3. Frustration score state machine

The escalation logic, drawn as what it actually is — a one-way ratchet with a trapdoor. 😅

```mermaid
stateDiagram-v2
    [*] --> Calm: session starts (score = 0.0)
    Calm --> Mildly_Annoyed: score += weight\n(e.g. "that's wrong" = +0.15)
    Mildly_Annoyed --> Frustrated: score += weight\n(e.g. "this is ridiculous" = +0.30)
    Frustrated --> Escalated: score ≥ 0.80 🆘
    Mildly_Annoyed --> Calm: no new signals\n(score never decreases,\nbut also never auto-cools)
    Calm --> Escalated: single strong signal\n(e.g. "connect me to a human" = +0.40,\ncan combine with prior turns)
    Escalated --> [*]: human_escalation_node\nticket issued, graph ends turn

    note right of Escalated
        Score is cumulative across the
        whole session (persisted via
        SqliteSaver, keyed by session_id).
        It only ever goes up within a turn —
        there's no decay mechanism today.
    end note
```

### 4. Middleware pipeline

Every request runs this gauntlet before it ever touches the agent graph:

```mermaid
flowchart LR
    Req(["Incoming request"]) --> RID["🏷️ RequestIDMiddleware\nassigns/echoes X-Request-ID"]
    RID --> RLM["🚦 RateLimitMiddleware\nsliding window per client IP"]
    RLM -->|"limit exceeded"| R429(["429 + Retry-After"])
    RLM -->|"under limit"| CORS["🌍 CORSMiddleware"]
    CORS --> TIME["⏱️ Request timing log\nmethod, path, status, duration_ms"]
    TIME --> ROUTE["Route handler\n(/ask, /ingest, /health...)"]
    ROUTE --> JSONLOG["📋 JSONLogFormatter\nstructured stdout for log aggregators"]
```

### 5. Concurrency control — the semaphore dance

Per-worker throttling so one uvicorn process doesn't fire 200 concurrent LLM calls at Anthropic:

```mermaid
flowchart TB
    subgraph Worker["Single uvicorn worker process"]
        SEM["asyncio.Semaphore(max_concurrent_llm_calls)"]
        R1["Request A"] -->|acquire| SEM
        R2["Request B"] -->|acquire| SEM
        R3["Request C"] -.waits.-> SEM
        SEM --> LLMCall["ChatAnthropic.ainvoke()"]
    end
    Note["Total concurrency across the fleet =\nmax_concurrent_llm_calls × num_workers\n(the semaphore is per-process, not global)"]
```

---

## 🧰 Tech Stack & Why

| Layer | Choice | Why 🤔 |
|---|---|---|
| **Orchestration** | LangGraph `StateGraph` | Explicit, inspectable state machine instead of an implicit agent loop — routing/escalation logic is testable as plain functions independent of any LLM call. |
| **LLM** | Claude via `langchain-anthropic` | One provider client reused for generation, routing, scope detection, and judging — swappable per-node by model name. |
| **Web framework** | FastAPI + Uvicorn | Native async matches LangGraph's async invocation path; free OpenAPI docs at `/docs`; first-class SSE for token streaming. |
| **Vector store** | Chroma, singleton + threading lock | Zero-ops embedded vector DB; singleton avoids N embedding-model loads and SQLite "database is locked" errors under concurrent requests. |
| **Embeddings** | HuggingFace BGE (`all-MiniLM-L6-v2`) locally, OpenAI opt-in | Local by default = no second paid API + no extra network hop on every retrieval. |
| **Keyword retrieval** | `rank-bm25` + custom `BM25Retriever` | Lexical fallback for exact-term queries dense embeddings under-rank. |
| **Hybrid fusion** | `EnsembleRetriever`, weights `[0.6, 0.4]` | RRF-style fusion of semantic + keyword without hand-rolling it. |
| **Doc parsing** | `unstructured[pdf,docx]` | Handles messy real-world PDFs/DOCX (tables, headers) better than naive text extraction. |
| **Chunking** | `RecursiveCharacterTextSplitter` | Splits on paragraph → line → sentence → word, preserving semantic units. |
| **Structured LLM output** | Pydantic + `.with_structured_output()` | Router intent, scope decision, judge scores are typed models — no hand-parsed JSON. |
| **Multi-turn state** | `SqliteSaver` checkpointer (fallback `MemorySaver`), keyed by `thread_id=session_id` | Frustration score + escalation flag survive a server restart now. |
| **Async task queue** | Celery + Redis | Crash-safe, retryable, horizontally-scalable ingestion — see the [ingestion diagram](#2-async-ingestion-via-celery--redis). |
| **CLI** | Typer + Rich | `ingest --file/-f` or `--dir/-d`, with `--dry-run` and a pretty summary table — nicer than raw `argparse` prints. |
| **Frustration detection** | Hand-written weighted regex bank, no LLM call | Deterministic, sub-millisecond, zero-cost per turn — this signal gates on *every single message*, so an LLM call here would be wasteful. |
| **Evaluation** | Custom LLM-judge + RAGAS | Two independent eval paths for faithfulness/relevance/completeness/hallucination and standardized retrieval metrics. |
| **Observability** | LangSmith (opt-in) + structured JSON logs + request IDs | Tracing when you want it, greppable/aggregatable logs always. |
| **Rate limiting** | In-memory sliding-window middleware | Simple per-process throttle; documented TODO to move to Redis `INCR` so multi-worker deployments share one limit. |
| **Tool exposure** | MCP server (stdio) | Any MCP client (Claude Desktop, etc.) can call `search_policies`/`list_policy_documents` directly — deliberately narrower than the guarded HTTP path. |
| **Packaging** | `pyproject.toml` optional-dependency groups (`api`, `mcp`, `eval`, `openai`, `dev`) | Install only what you need. |
| **Linting/typing** | `ruff` + `mypy --strict` | Fast lint/import-sort + strict typing on a `TypedDict`-heavy state model. |

---

## 💡 Core Design Decisions

### 🛡️ Guardrails live *in* the graph, not in HTTP middleware
`input_guard` and `output_guard` are LangGraph nodes, so the same checks apply whether you hit the agent over HTTP, embed it directly in Python, or (partially) via MCP. Trade-off: a blocked request still costs one node execution — there's no free "reject before graph starts" path.

### 😤 Frustration is cumulative and one-directional
`update_frustration_score()` never subtracts — once you're annoyed, the score only climbs within the session (see the [state machine](#3-frustration-score-state-machine)). Crossing `ESCALATION_THRESHOLD = 0.80` routes to `human_escalation_node` **regardless of what the router would've decided** — a deterministic, regex-computed circuit breaker sitting on top of an otherwise LLM-driven pipeline.

### 🧭 Structured routing that fails open
`router_node` classifies intent via `.with_structured_output(RouterDecision)`. If the router call throws, the code **defaults to `"policy"`** rather than failing the whole request — availability over strict scope enforcement, by explicit design.

### 🥈 Hybrid retrieval is opt-in
`build_retriever()` only builds the BM25 + semantic ensemble when `use_bm25=True` **and** a corpus is supplied. It's there and ready, but wiring a corpus into `retrieve_node` at call time is what flips it on.

### 🧵 Output grounding via cheap lexical overlap, not embeddings
`_is_grounded()` tokenizes retrieved context + generated response (words >4 chars), and flags low grounding if a >20-word response shares <3 words with the retrieved vocabulary. Deliberately lightweight — a full embedding-similarity check would cost an extra model call on *every* response.

### 📦 Celery + Redis over fire-and-forget background tasks
Documented directly in `ingest.py`'s docstring — the switch bought crash recovery, real status polling, automatic retry with backoff, horizontal scaling, and process isolation (the old approach shared the API's event loop; the new one runs in a totally separate worker process).

### 🔒 Chroma as a locked singleton
`get_vectorstore()` uses a double-checked lock so 20 concurrent requests share one Chroma client instead of spinning up 20 (20x the embedding-model RAM, 20 SQLite file handles, guaranteed "database is locked" errors).

### 🚦 Per-process semaphore + timeout around every LLM call
`/ask` acquires an `asyncio.Semaphore(max_concurrent_llm_calls)` before invoking the graph, wrapped in `asyncio.wait_for(agent_timeout_seconds)`. The semaphore is explicitly **per-worker-process** (documented in `dependencies.py`) — total fleet concurrency is `max_concurrent_llm_calls × num_workers`, so the per-worker value should be tuned against your actual Anthropic rate limit.

### 🩺 Two-tier health checking
`/health` is a pure liveness probe for pod-restart decisions. `/status` actually opens the Chroma collection and reports `degraded` (not a 5xx) if it can't — because the service can still limp along answering from parametric knowledge even with retrieval down.

### 🔌 MCP server as a narrower surface
`mcp_server.py` calls the retriever directly, skipping guardrails and frustration tracking entirely — MCP tools are meant to hand raw context to an *external* LLM's own reasoning loop, not re-expose the fully-guarded conversational agent.

---

## 📁 Module Map

```
ragent/
├── main.py                          # uvicorn entrypoint
├── scripts/
│   └── ingest_data.py                 # 🖥️  Typer CLI: --file/--dir, --clear, --dry-run, Rich summary table
└── src/
    ├── agents/
    │   ├── state.py                   # AgentState TypedDict — shared schema across every graph node
    │   ├── master_agent.py            # top-level StateGraph: guardrails, frustration, routing, escalation
    │   ├── policy_agent.py            # sub-graph: transform_query → retrieve → generate
    │   └── __init__.py
    ├── api/
    │   ├── app.py                      # FastAPI factory (lifespan, middleware wiring, routers)
    │   ├── middleware.py               # 🆕 RequestID, RateLimit (sliding window), JSON log formatter
    │   ├── dependencies.py             # verify_api_key, get_llm_semaphore, get_request_id
    │   ├── schema.py                   # Pydantic request/response models
    │   └── routes/
    │       ├── query.py                # POST /ask, POST /ask/stream (SSE) — semaphore + timeout wrapped
    │       ├── ingest.py                # POST /ingest (Celery), GET /ingest/status/{task_id}
    │       ├── ingest_bgtask.py         # 🗄️ archived: the old BackgroundTasks version, kept for reference
    │       └── health.py                # GET /health (liveness), GET /status (readiness)
    ├── worker/
    │   ├── celery_app.py                # 🆕 Celery config: JSON serialization, late acks, single-prefetch
    │   └── tasks.py                     # 🆕 ingest_documents task — retry w/ backoff, progress reporting
    ├── guardrails/
    │   ├── input_guard.py                # PII (CC/SSN) + prompt-injection regex filter
    │   ├── output_guard.py               # PII redaction + lexical grounding heuristic
    │   ├── frustration_detector.py       # weighted regex signal bank + cumulative score
    │   └── scope_detector.py             # standalone LLM in/out-of-scope classifier
    ├── rag/
    │   ├── document_processor.py         # UnstructuredFileLoader + RecursiveCharacterTextSplitter
    │   ├── embeddings.py                  # local (HF BGE) or OpenAI embedding provider, lru_cached
    │   ├── vectorstore.py                 # 🆕 Chroma singleton (thread-locked) + retriever factory
    │   └── retriver.py                    # BM25Retriever + EnsembleRetriever hybrid builder
    ├── evaluation/
    │   ├── judge_llm.py                   # structured-output LLM judge: 4-dimension scoring
    │   └── ragas_eval.py                   # RAGAS harness (faithfulness, answer relevance, context precision)
    ├── observability/
    │   └── tracing.py                      # LangSmith env-var setup, run URL builder
    ├── mcp_server.py                        # stdio MCP server: search_policies, list_policy_documents
    └── config/
        └── settings.py                      # pydantic-settings, .env-backed
```

---

## ⚙️ Configuration Reference

All settings load via `pydantic-settings` from environment variables / a `.env` file (case-insensitive).

| Setting | Default | Notes |
|---|---|---|
| `anthropic_api_key` | `""` | Required for any LLM call to succeed |
| `llm_model` | `claude-opus-4-8` | Shared across generation, routing, scope detection, judging |
| `llm_max_tokens` / `llm_temperature` | `4096` / `0.1` | Low temperature favors grounded answers over creativity |
| `embedding_provider` / `embedding_model` | `local` / `all-MiniLM-L6-v2` | `local` (HF BGE, CPU) or `openai` |
| `chroma_presist_dir` / `chroma_collection_name` | `./data/chroma-db` / `policies` | Persistent vector store location |
| `retriver_top_k` | `6` | Top-k for retrieval |
| `use_bm25` | `False` | Enables hybrid retrieval once a corpus is wired in |
| `chunk_size` / `chunk_overlap` | `100` / `200` | Splitter config |
| `langsmith_api_key` / `langsmith_project` / `langsmith_tracing` | — / `ragent` / `False` | `langsmith_enabled` gates on both key + flag |
| `redis_url` | `redis://localhost:6317` | 🆕 Celery broker + result backend |
| `max_concurrent_llm_calls` | `10` | 🆕 Per-worker semaphore size |
| `agent_timeout_seconds` | `60` | 🆕 Hard timeout around `master_agent.ainvoke()` |
| `rate_limit_per_minute` | `60` | 🆕 Sliding-window limit per client IP |
| `cors_origin` | `["*"]` | 🆕 CORS allow-list |
| `api_secret_key` | `""` | 🆕 Bearer token for `verify_api_key` — unset = auth disabled |

---

<div align="center">

*Built as a hands-on exploration of guardrail-first agent design — issues, PRs, and "hey this regex is wrong" comments welcome.* ✨

</div>
