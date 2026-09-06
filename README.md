# FaceID Chain — Face Identification & Blockchain Verification

A pipeline that takes a **face scan as input**, **identifies matching content on
the web / social media**, and then **verifies that discovered data on a
blockchain** — end to end.

```
Face scan input  →  Web / social media search (find matching post)  →  Blockchain upload & verification
```

- **Face identification** — YuNet ONNX face detection + **SFace** 128-d face
  embeddings (via OpenCV `cv2.FaceDetectorYN` / `cv2.FaceRecognizerSF`) produce a
  real neural face fingerprint used for same-person matching (cosine similarity).
- **Web / social media search** — a *genuine* search runs on the uploaded image:
  - **Bing Visual Search (reverse-image)** — the image is uploaded to Bing and
    visually-similar images (with their originating page URLs) are returned;
    every candidate is **face-matched** against the input so only real matches
    survive.
  - **Text search enrichment** — a text query (Google Custom Search JSON API when
    `GOOGLE_API_KEY`/`GOOGLE_CSE_ID` are set, else Bing) finds candidate posts;
    each result page is fetched and its images (og:image / `<img>`) are
    **face-confirmed** before the page counts as a matching post.
- **Blockchain verification** — once a post matches, its fingerprint (URL, title,
  snippet, match score, hashes, metadata) is **mined into a hash-chained ledger**.
  A re-verification step re-derives the post hash, confirms Merkle membership, and
  recomputes the whole chain hash-linkage to prove the record is tamper-evident.

A small web UI (served by FastAPI) lets you upload a face, watch the 3-step
pipeline run, and inspect the on-chain record. A CLI is also included.

---

## Pipeline shape

```
┌──────────────┐     ┌───────────────────────────┐     ┌─────────────────────────────┐
│  1 · Face    │     │  2 · Web / social search  │     │  3 · Blockchain verify      │
│   detection  │     │  (Google, genuine search) │     │  (local hash-chained chain) │
│  + encoding  │ ──► │  candidates face-matched  │ ──► │  record mined + re-verified │
└──────────────┘     └───────────────────────────┘     └─────────────────────────────┘
```

1. **Detect & encode** — `app/face/matcher.py` detects faces (YuNet, Haar
   fallback) and encodes each into a 128-d **SFace** embedding (perceptual
   hash/colour fallback if the model is absent) → a `face_id`.
2. **Search** — `app/search/webfinder.py` performs a **genuine** search:
   - **Bing Visual Search** reverse-image query on the input image (no key).
   - **Text search** via Google CSE JSON API (if configured) or Bing, then page
     images are extracted and **face-matched** against the input (SFace cosine ≥
     model threshold). Only genuine same-face matches are kept.
3. **Blockchain** — `app/blockchain/chain.py` implements a local, persisted,
   tamper‑evident chain: SHA‑256 chained blocks, Merkle roots, difficulty‑bounded
   proof‑of‑work mining, and a public `verify_record(...)` API. The winning post
   fingerprint is mined in; the pipeline then **re-verifies** it on‑chain.

---

## Repo layout

```
app/
  face/matcher.py          YuNet + SFace detection/encoding + matching
  search/webfinder.py      genuine Bing reverse-image / Bing·Google text search + face-matching
  blockchain/chain.py      local hash-chained blockchain (JSON ledger)
  pipeline/pipeline.py     end-to-end orchestrator
  web/server.py            FastAPI app + REST endpoints
  web/static/index.html    web UI
  cli.py                   command-line interface
scripts/
  download_models.py       fetch YuNet + SFace OpenCV models
  make_sample.py           generate a synthetic "face" image for offline testing
tests/
  test_chain.py            tamper-evidence + record verification tests
  test_face.py             face detection / matching tests
main.py                    Uvicorn entrypoint (Render / production)
render.yaml                Render blueprint
data/ledger.json           running ledger (created on first write, gitignored)
```

> `models/`, `data/*.json` and `.venv` are gitignored. `scripts/download_models.py`
> fetches the two small model files (~39 MB) at setup/build time.

---

## Requirements

- Python 3.10+ (developed/tested on 3.13)
- Tested on Windows 11 and Ubuntu (Render)

## Setup (local)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# download the YuNet + SFace ONNX models (~0.2 MB + ~38 MB) for real face matching
python scripts/download_models.py

# optional but recommended: real Google API search
cp .env.example .env               # then fill GOOGLE_API_KEY and GOOGLE_CSE_ID
```

### Run the web app

```bash
uvicorn main:app --reload --port 8000
# open http://localhost:8000
```

REST API:

| Method | Endpoint          | Description                                   |
|--------|-------------------|-----------------------------------------------|
| GET    | `/`               | Web UI                                        |
| GET    | `/api/health`     | Service + chain status                        |
| POST   | `/api/pipeline`   | Run full pipeline (multipart `image` upload)  |
| GET    | `/api/verify?data_hash=...` | Re-verify an on-chain record        |
| GET    | `/api/chain`      | Dump the ledger + tamper check                |

### Run the CLI

```bash
python -m app.cli samples/photo.jpg --query "Jane Doe" --platform instagram.com
python -m app.cli samples/photo.jpg --json
```

### Offline smoke test (no real photo needed)

```bash
python scripts/make_sample.py            # writes samples/sample_face.png
python -m app.cli samples/sample_face.png --json
```

### Run tests

```bash
python -m pytest tests -v                # or: python tests/run_tests.py
```

---

## Blockchain details

- **Blockchain used:** a **local, simulated blockchain** (JSON ledger at
  `data/ledger.json`), no external network/keys required. Suitable for
  demonstration, CI, and Render.
- **Structure:** each block contains `index`, `timestamp`, `previous_hash`,
  `difficulty`, `nonce`, `transactions`, `merkle_root`, `hash`.
- **Chaining:** block *i*'s hash includes block *i−1*'s hash, so changing any
  earlier block invalidates every later block.
- **Merkle root:** each block's transaction hashes are merged into a single root
  that is bound into the block hash.
- **Proof of work (simulated):** blocks are mined against a difficulty target
  (default 3 leading zero hex characters). Mining demonstrates the cost of
  producing a valid block and is configurable via `BLOCKCHAIN_DIFFICULTY`.
- **Record hash:** the confirmed post is fingerprinted as `data_hash =
  SHA-256({post_url, title, snippet, source, match_score, matched_face_id,
  url_hash, input_image_hash, discovered_at})`, stored as a transaction.
- **Re-verification (`verify_record`):** re-derives the data hash, confirms the
  transaction exists on-chain, checks the Merkle membership, and (via
  `verify_tamper`) recomputes the entire block-hash linkage. Only a record that
  is genuinely on-chain and unmodified returns `verified: true`.

> **Why a local chain?** Every blockchain (public or local) provides the same
> tamper-evident record property: a hash-chained, Merkle-rooted append-only
> ledger. A local chain makes the project fully self-contained and runnable
> anywhere. Swapping in a public testnet (Sepolia, goerli, etc.) only requires
> replacing `LocalChain` with a thin RPC wrapper — the fingerprint/verification
> logic is identical.

---

## Deployment (Render)

The repo ships a `render.yaml` blueprint:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

Or, manually with the Blueprint:
1. In [Render](https://render.com) Dashboard → **New +** → **Blueprint**.
2. Point it at this GitHub repo; Render will create a free **web service** with:
   - Build: `python scripts/download_models.py && pip install -r requirements.txt`
   - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Health check: `/api/health`
   - A 1GB disk mounted at `/data` holding the persisted ledger
     (`BLOCKCHAIN_LEDGER_PATH=/data/ledger.json`).
3. Deploy, open the URL, upload a face image, run the pipeline.

> The web search step calls Google from Render's servers. For the most reliable
> results set `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` as Render env vars.

---

## Known limitations

- **Face matching depends on model availability:** YuNet + SFace give accurate
  detection and recognition, but the ONNX model files are downloaded by
  `scripts/download_models.py` (or at Render build time). Without them the code
  falls back to whole-frame + perceptual hash matching, which is weaker.
  Older/artistic or very grainy photos may occasionally fail detection.
- **Web search reliability:** search engines rate-limit and change their
  scraping contract over time. Bing reverse-image and Bing text scraping are
  genuinely functional but unauthenticated and can occasionally return fewer /
  server-substituted results. The Google Custom Search JSON API backend is the
  stable, recommended option when a key is configured.
- **Reverse-image vs face matching:** Bing returns *visually similar* images;
  the pipeline **face-verifies** each one with SFace, so only posts that
  genuinely contain the same face are reported. Different people in similar
  photos are correctly rejected. A photo of someone whose face is not indexed
  anywhere online will (correctly) produce no matches.
- **Local chain persistence:** on Render the ledger is persisted to the mounted
  `/data` disk; unsaved free-instance restarts can still re-seed the genesis
  block (new chain). In local dev the ledger persists across runs.
- **Mining cost:** difficulty 3 is fast (ms). Higher difficulty = stronger PoW
  proof but slower writes.
- **No OCR:** the search can be biased with a text hint (`query` param) but text
  is not auto-extracted from the input image.

## Getting real results

1. Use a clear, front-facing photo of one person.
2. Add the person's name/username as the **search hint** (`query`) — this guides
   the text-search enrichment, and the final result is still *face-verified*, so
   a wrong hint won't produce a false match.
3. The person's face must be indexed somewhere on the web (Wikipedia, news,
   social profiles). The demo/test images under `samples/` are automatically
   gitignored.

## License

MIT — see [LICENSE](LICENSE).