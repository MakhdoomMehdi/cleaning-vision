# Cleaning Vision AI

Local MVP that compares reference images of a property location with images uploaded after a
cleaning job. YOLO (local) localizes objects; OpenAI Vision analyzes differences; a validation
step decides what is reported. **Status: comparison API + dashboard + optional OpenAI vision.**

## Architecture (target)

```text
React Frontend -> FastAPI -> Upload & Validation -> Preprocessing -> YOLO Detection
  -> Object Matching & Viewpoint Assessment -> OpenAI Vision Comparison
  -> Finding Validation -> SQLite + local image storage -> JSON + annotated images -> Dashboard
```

A deterministic Python orchestrator calls independent services; model providers
(`providers/yolo_provider.py`, `providers/openai_provider.py`) are kept separate from business
logic. No LangGraph, agents, Redis, Celery, or Docker.

Key rules: an unmatched detection is never proof an object is missing; the OpenAI model never
produces authoritative bounding boxes (boxes come only from the detector); the system never
declares a property "clean" from limited images.

## Requirements

- Python 3.11+ (developed on 3.12)
- Node.js 20.19+ and npm

## Setup and run

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                # macOS/Linux: cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Check: <http://localhost:8000/health> (interactive docs at `/docs`).

Set `OPENAI_API_KEY` in `backend/.env` (never commit it). `OPENAI_MODEL` defaults to `gpt-5.4`;
change it if your account does not have access to that model.

### Frontend

```powershell
cd frontend
npm install
copy .env.example .env                # optional; defaults to http://localhost:8000
npm run dev
```

Open <http://localhost:5173>. Enter a property id, upload a **before** photo, then upload the
**after** photo and click Compare. Boxes are drawn only on flagged issues (not on every object).
Stains, debris and damage need `OPENAI_API_KEY` in `backend/.env`. Sample pairs:
`images/before1.jpg` + `after1.jpg` (bedroom) and `before2.jpg` + `after2.jpg` (garage).

## Object detection (Phase 2)

`POST /api/v1/detect` (multipart field `file`; optional query `conf_threshold`, `iou_threshold`)
returns:

```json
{ "image_id": "img_001", "width": 1280, "height": 720,
  "detections": [{ "id": "det_001", "label": "chair", "confidence": 0.94,
                   "bbox": [120, 150, 350, 500] }],
  "model": { "name": "yolo11n.pt", "conf_threshold": 0.25, "iou_threshold": 0.45, "device": "cpu" },
  "note": "..." }
```

- `bbox` is `[x1, y1, x2, y2]` in **original-image pixels** after applying EXIF orientation, so
  it matches what a browser displays. `confidence` is the raw model score.
- `GET /api/v1/detect/classes` lists the 80 COCO classes the pretrained model knows. Anything
  else (a toolbox, a wall scuff) is *unsupported*, not *absent*. An empty `detections` list means
  "no supported object found", never "the scene is empty".
- The model (`yolo11n.pt`) is loaded lazily on first request and reused. Weights are downloaded
  automatically by Ultralytics on first use (needs internet once). Inference runs on CPU.
- Errors: `413` too large, `415` not a decodable JPEG/PNG/WebP (content is checked, not the
  extension), `422` bad/corrupt/out-of-range dimensions, `503` detector unavailable.

### Sample-image workflow

`backend/tests/sample_images/cats_couch.jpg` is a COCO val2017 photo (id 39769; COCO images carry
Flickr licenses, CC-BY 2.0 family; used here only as a local test sample). Try it:

```powershell
curl.exe -X POST -F "file=@tests/sample_images/cats_couch.jpg" http://localhost:8000/api/v1/detect
```

Observed on this machine (CPU): 2 cats, 1 remote, 1 couch (the second remote was missed). First
request ~12 s including model load, later requests ~0.2 s. This is a single anecdotal check, not
an accuracy evaluation.

## Reference images (Phase 3)

In the UI: enter a **property identifier**, pick a **location**, and upload one or more reference
images. Saved images for that property/location appear under *Available reference images*.

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/locations` | Supported locations: garage, bedroom, kitchen, bathroom, ceiling, room_corner |
| `POST /api/v1/references` | Multipart: `property_id`, `location`, one or more `files`. Returns `201` and the saved items |
| `GET /api/v1/references` | List; optional `property_id` and `location` filters |
| `GET /api/v1/references/{id}/image` | Serves the stored image (original bytes) |

- `property_id` is a plain label: letters, digits, `_ . -`, starts alphanumeric, max 64 chars.
- An upload is **all-or-nothing**: every file is validated first (decoded content, size,
  dimensions); if any fails nothing is stored (`413` too large, `415` not an image, `422` other).
  If the database write fails, already-written files are deleted again.
- Files are stored on disk under server-generated names (`uploads/references/<uuid>.jpg`). The
  user's filename is kept only as sanitised display metadata and is never used in a path; the
  property id is never used in a path either. Stored keys are re-validated on every read/delete.
- Metadata (UUID id, property, location, dimensions, size, SHA-256, UTC timestamp) is in SQLite
  (`backend/data/cleaning_vision.db`, created automatically; relative `DATABASE_URL` /
  `UPLOAD_DIR` paths resolve against `backend/`). Only the `reference_images` table exists so
  far; the other tables are added in the phases that use them.
- Storage sits behind an `ImageStorage` interface (`save` / `read` / `delete` by key), and
  database access goes through SQLAlchemy, so S3 and PostgreSQL can replace them later.

## Before/after comparison (Phase 4)

`POST /api/v1/inspections` (multipart: `property_id`, `location`, 1+ `files`) runs the pipeline
synchronously and returns `201` with the stored inspection. There is no `queued` status: the
result is `completed` or `failed` when the response arrives.

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/inspections` | Upload inspection images and run the comparison |
| `GET /api/v1/inspections` | History; filters `property_id`, `location`, `limit`, `offset` |
| `GET /api/v1/inspections/{id}` | Inspection, its images and per-image detections |
| `GET /api/v1/inspections/{id}/results` | Comparison: alignment, coverage, matches, candidates |
| `GET /api/v1/inspections/{id}/images/{ref_N\|new_N}` | Stored reference / inspection image |

How images are compared (`matching-v1`, deterministic, no model calls):

1. **Viewpoint:** SIFT features, ratio-test matching, RANSAC homography. An alignment counts only
   with at least 15 inliers, an inlier ratio of 0.25 and a plausible transform. The result is
   `aligned`, `different_viewpoint` or no reliable alignment (`inconclusive`).
2. **Object matching:** same-label detections are paired by box overlap, then by feature support
   inside the box (so a detector miss is not read as a removal), then by colour appearance
   (`possible_moved`). An unmatched detection is never treated as proof.
3. **Leftover reference objects:** outside the new view -> `not_visible`; mostly covered by a
   different object -> `possibly_occluded`; otherwise `possible_missing`. New objects the
   reference view cannot account for are `possible_new`.
4. **Coverage:** each reference image is rated `full` / `partial` / `none` by how much of it the
   inspection images show. If coverage is `insufficient`, the result says so and asks for an
   additional image instead of reporting a clean room.

Candidates are *candidates for review*, not findings. `evidence_strength` is `low` or `medium`,
derived from rules (alignment quality, inlier count, detector confidence, image quality); it is
not a probability. A missing object has no "after" box, because none exists. Image quality
(dark, overexposed, blurry) is reported alongside.

Failures are recorded, not hidden: if a stage (`load_images`, `detect_objects`, `compare_images`,
`save_results`) fails, the inspection is stored with `status: failed`, `failed_stage` and a safe
`error_message`. Bad input is rejected with 4xx and nothing is stored; `409` if the location has
no reference images; `422` if it has more than `MAX_REFERENCE_IMAGES_PER_INSPECTION` (never
silently truncated).

Sanity check with the real model on one sample photo with controlled edits: a camera zoom/rotation
alone produced no candidates; painting one cat out produced a single `possible_missing` cat; moving
the camera so the cats left the frame produced `not_visible` and a request for another image. These
are three anecdotal cases (`tests/test_real_pipeline_integration.py`, opt-in), not an accuracy
evaluation.

## Tests and checks

```powershell
cd backend ; .\.venv\Scripts\python.exe -m pytest
cd frontend ; npm test ; npm run build
```

Backend tests use a temporary SQLite database and upload folder per test and a fake detector, so
they need no model weights and never touch your real data. To also run the real YOLO
model: `$env:RUN_YOLO_INTEGRATION = "1"; .\.venv\Scripts\python.exe -m pytest tests/test_yolo_integration.py tests/test_real_pipeline_integration.py`.

## Configuration

All settings are environment variables; see `backend/.env.example` and `frontend/.env.example`.
The OpenAI key is read only by the backend and is never sent to the frontend or logged.

## Licensing note (must read before commercial use)

Ultralytics YOLO is licensed under **AGPL-3.0** (or a paid Ultralytics
Enterprise license). Free for local experimentation, but deploying it in a network service or
commercial product has source-disclosure obligations. Review the license or obtain an enterprise
license before any commercial deployment.

## Security notes

The local MVP has **no authentication**. Before any deployment you must add authentication,
property-level authorization, and private (non-public, access-controlled) image access, and
move to HTTPS, PostgreSQL, and object storage such as S3.

## Known limitations (so far)

- Health, detection, reference management and the deterministic comparison (Phases 1-4) exist.
  There is no OpenAI analysis, human review or final decision yet (Phases 5-6), and no
  inspection UI.
- Comparison candidates are stored as JSON on the inspection; separate findings/review tables
  come with Phase 5/6.
- Alignment uses a single homography, which is only accurate for roughly planar scenes or small
  camera moves. Strong parallax can cause wrong "moved"/"not visible" calls, and feature matching
  needs texture (blank walls often give no reliable alignment, which yields `inconclusive`).
- Coverage uses the best-matching inspection image per reference image; it does not combine
  several images into a mosaic.
- No way to delete or replace a reference image yet, and no de-duplication of identical uploads.
- Schema is created with `create_all`; there are no migrations (fine for the MVP, add Alembic
  before PostgreSQL).
- Pretrained YOLO11n covers 80 generic COCO classes only; it cannot detect most
  property-specific objects or damage.
- Dimensions are validated only by the backend (client checks type and size).
- Nothing here has been evaluated for accuracy, and it is not production-ready.
