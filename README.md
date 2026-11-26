# Cleaning Vision AI

Compares saved before photos of a room with one photo taken after a guest stay. OpenAI looks at
the whole before pool, uses the angles that match the after photo, and reports **damage**,
**missing**, and **new**. The result shows the matched before photo, the after photo, and a
third findings photo with one color per finding. The box has no label; the list under the photo uses that same color.

## Architecture

```text
React -> FastAPI -> save the after photo -> load the before pool
  -> one OpenAI vision call -> text report -> SQLite + local image files
```

The model is told that a different camera angle is not proof something is missing. If the after
photo shows a part of the room that was never photographed, the result is inconclusive and the
three lists stay empty. The report never certifies that a room is clean.

## Requirements

- Python 3.11+ (developed on 3.12)
- Node.js 20.19+ and npm
- An OpenAI API key. Comparison does not run without it.

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

Set `OPENAI_API_KEY` in `backend/.env` (never commit it). `OPENAI_MODEL` defaults to `gpt-5.4`.
Check <http://localhost:8000/health>.

### Frontend

```powershell
cd frontend
npm install
copy .env.example .env                # optional; defaults to http://localhost:8000
npm run dev
```

Open <http://localhost:5173>. Enter a property id, upload one or more **before** photos of the
room (different angles are fine), then upload the **after** photo and click Compare.

## What an inspection does

`POST /api/v1/inspections` (multipart: `property_id`, `location`, one after image) runs
synchronously and returns `201`. The result is `completed` or `failed` when the response arrives.

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/references` | Save before photos for a property and location |
| `POST /api/v1/inspections` | Upload the after photo and run the comparison |
| `GET /api/v1/inspections/{id}/results` | Damage, missing, and new, as text |
| `GET /api/v1/inspections/{id}/images/{ref_N\|new_N}` | A stored before or after image |

The before pool for one inspection is capped by `MAX_REFERENCE_IMAGES_PER_INSPECTION` (default
12). More than that is rejected, not silently dropped. A location with no before photos is
checked on its own for guest readiness: damage, cleanliness, and anything that is not ready
for the next guest, each with a box and a comment.

## Tests

```powershell
cd backend ; .\.venv\Scripts\python.exe -m pytest
cd frontend ; npm test ; npm run build
```

Backend tests use a temporary database and a fake vision client, so they do not call OpenAI.

## Security notes

The local MVP has **no authentication**. Before any deployment you must add authentication,
property-level authorization, and private image access, and move to HTTPS, PostgreSQL, and
object storage such as S3. The OpenAI key is read only by the backend.
