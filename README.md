# 3DGS Visual Localization with AI Copilot

A research-oriented web platform for estimating a query camera pose in a COLMAP reconstruction, reviewing saved results, inspecting Gaussian PLY scenes, and asking a data-grounded AI copilot about the workspace.

The interface is designed around the user's result: upload a clear photo, choose a scene, and review the estimated position and whether it needs checking. Matching counts and pixel-level alignment details are available under “Détails supplémentaires”. Photos that are almost entirely black or have no visible detail are stopped with an actionable message; localization acceptance thresholds are unchanged.

The original Colab notebooks and large scene assets are left untouched. The app's localization path uses SIFT query/reference descriptors linked to **observed COLMAP 3D points**, then estimates a world-to-camera pose using OpenCV PnP-RANSAC. DINOv2 retrieval is available as an optional dependency; when it is off, the candidate prefilter is a color-histogram fallback and is explicitly reported in each run.

The scene explorer embeds the official SuperSplat Viewer on the PlayCanvas engine for the original binary 3DGS PLY. It renders the anisotropic Gaussians, quaternion rotations, sigmoid opacity, base spherical-harmonic color and all 45 higher-order coefficients (SH degree 3). The PLY is served unchanged; localization camera overlays are rendered in the same COLMAP world frame when the configured alignment flag is enabled. Initial framing uses the registered COLMAP camera nearest the trajectory median (`GET /api/scenes/{scene_id}/viewer-camera`), with its own calibrated vertical field of view, rather than averaging directions across the route. This is not a point-cloud or mesh approximation.

## Status and scientific limits

- The notebook documents COLMAP poses as `X_camera = R @ X_world + t`. Stored app poses follow the same world-to-camera convention; the camera center is `-R.T @ t`.
- Query intrinsics default to the notebook's `fx=fy=716.614` at `1242×375`. They are scaled independently in x/y to the uploaded image and the working image size. Set all `QUERY_*` values for the actual query camera; the application does not average the 447 COLMAP focal lengths.
- The query calibration currently assumes a pinhole camera with no distortion coefficients. Verify calibration and cropping/aspect ratio for uploaded images.
- Translation is in COLMAP scene units. The platform does not assume those units are metres.
- A numerical coordinate-overlap check of the configured road assets found 471,603 PLY Gaussian centers and 99,150 COLMAP points; 99.94% of COLMAP points fall inside the PLY axis-aligned bounds, and nearest-center distances are ≤0.05 COLMAP units for 94.59% of COLMAP points and ≤0.5 units for 99.73%. Together with the notebook's matching reference render, this supports use of a shared frame for this scene. The flag `SCENE_FRAME_VERIFIED=1` enables its camera overlay. This does not verify metric scale or establish one-to-one point correspondence.
- The notebook's rendered-depth 2D–3D correspondence method remains experimental. This app's active method uses image observations associated with COLMAP 3D points; it has not been benchmarked against the notebook's preliminary 45-image evaluation.
- A `localized` status means this run crossed the configured inlier, inlier-ratio, and reprojection thresholds. It is not a validated accuracy claim.
- DINOv2, PyTorch, COLMAP model files, reference images, and an LLM API key are external/configurable. Missing inputs produce explicit unavailable/fallback states rather than fabricated poses or responses.

## Project layout

```text
backend/
  app/
    localization/    # COLMAP loading, camera geometry, SIFT + PnP engine
    routes/          # Flask REST API
    services/        # scene discovery, persistence workflow, copilot
    models.py        # SQLAlchemy localization model
  tests/
  requirements.txt
  requirements-ml.txt
  run.py
frontend/
  src/
    App.tsx           # dashboard, localization, explorer, history, copilot
    SceneViewer.tsx   # SuperSplat/PlayCanvas renderer and COLMAP pose overlays
    api.ts
```

## Requirements

- Python 3.10–3.12 recommended (for available PyTorch/pycolmap wheels).
- Node.js 20+ and npm.
- PostgreSQL is supported. SQLite is the local-development default.
- A COLMAP sparse reconstruction and its registered reference images. A Gaussian PLY is optional for localization and required only for point-cloud display.
- Optional: PyTorch/DINOv2 weights for global DINO retrieval.
- Optional: an OpenAI-compatible API key for generated Copilot responses.

## Configure scene data

Each scene directory should look like:

```text
data/scenes/road/
  sparse/0/                 # COLMAP model: cameras/images/points3D (.bin or .txt)
  images/                   # reference images, matching COLMAP image names
  scene.ply                 # optional 3D Gaussian scene
```

The app also checks `sparse/` when the model has no `0/` subfolder. Set `REFERENCE_IMAGES_DIR` if reference images are stored outside each scene directory. For assets stored in separate locations, configure `COLMAP_MODEL_PATH`, `GAUSSIAN_PLY_PATH`, and `REFERENCE_IMAGES_DIR` as absolute paths in `backend/.env`.

The supplied PLY is next to the source repository and is configured directly in `backend/.env`; it is not copied into Git or duplicated in the external data folder. The notebook's saved Colab outputs confirm the road COLMAP model, 447 reference images, and `query.jpg` in Google Drive. They must be downloaded from Drive to run the full pipeline locally. `tandt_db.zip` contains four different scenes (train, truck, drjohnson, playroom); do not pair their reconstructions with the road PLY.

The configured PLY-only scene is visible in the 3D explorer, but it is intentionally marked **Setup required** for localization until the matching road COLMAP model and reference images are supplied.

### Download and organize the Google Drive dataset

Google Drive does not need to be installed or mounted. In the Google Drive browser, download the dataset folders/files and extract them into the external `data/` directory, which is a sibling of this source repository:

```text
projet/
  data/                                      # outside rendu/, ignored by Git
    kitprojet/
      3DGS/
        sparse/
          0/
            cameras.bin
            images.bin
            points3D.bin
        data/                                # all 447 reference images, unchanged
      data/
        query.jpg
```

In the Drive browser, open `kitprojet/3DGS/sparse/0` and download only `cameras.bin`, `images.bin`, and `points3D.bin`. Keep these three files together in the same `0` folder. Then download the complete `kitprojet/3DGS/data` folder containing the 447 reference images, and `kitprojet/data/query.jpg`. Place the 447 image files directly inside `3DGS/data` (the Drive folder contains names such as `0000000000.png`); preserve their names exactly as listed in COLMAP's `images.bin`. Do not rename them, add another directory level, or mix `query.jpg` into the reference folder. If Drive downloads a ZIP, extract its contents into the exact paths above, avoiding extra nesting such as `sparse/0/0` or `data/data`.

For this app, `rigs.bin`, `frames.bin`, and `calib.txt` are not required downloads. The localizer reads the three core sparse files; query intrinsics already have a configured fallback (`716.614 716.614 621 187.5` at `1242x375`). The road PLY is already configured beside the project, so do not download another PLY or any files from `tandt_db.zip`.

The local `.env` already points `DATA_ROOT` to `projet\data` relative to the backend. Its paths are derived from that root. If you choose another download location, edit `DATA_ROOT` in `backend/.env`. You can instead set individual paths with `COLMAP_MODEL_PATH`, `REFERENCE_IMAGES_DIR`, `QUERY_IMAGE_PATH`, `QUERY_CALIB_PATH`, and `GAUSSIAN_PLY_PATH`. Relative `DATA_ROOT` paths are resolved relative to `backend/`, not the current shell directory. `.env` and `/data/` are excluded from Git. Restart Flask after changing paths.

Check `GET /api/data-status`: it reports resolved paths, COLMAP file presence, the reference-image count, and query-image availability. The dashboard displays the same checklist. The expected essentials are `cameras`, `images`, and `points3D` all `true`, `reference_image_count` equal to 447, and `query_image_available` equal to `true`. `rigs`, `frames`, and `query_calib_available` may be `false`. Once the scene is ready and the query is present, the Localization page offers **Localize configured query.jpg**; it uses the real image and never creates a placeholder pose.

## Start the backend (PowerShell)

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# The existing .env points DATA_ROOT at ..\..\data; change it if you downloaded elsewhere.
python run.py
```

The API is served at `http://127.0.0.1:5000/api`. The first localization request indexes SIFT features and their nearest COLMAP observations; allow extra time for the first run.

To use PostgreSQL, set for example:

```text
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/visual_localization
```

Install the PostgreSQL driver separately with `python -m pip install psycopg[binary]`. SQLAlchemy creates the localization table at backend startup.

To enable DINOv2 retrieval, install `requirements-ml.txt`, set `ENABLE_DINOV2=1`, and ensure the DINOv2 repository/weights can be loaded by `torch.hub`. Without this option, the application reports and uses a color-histogram candidate prefilter.

For AI responses, set `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`. Without a key, the Copilot gives a clear configuration notice and live database summary. Tool calling is restricted to retrieval functions and localization with an attached image after an explicit user request.

## Start the frontend

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api` to the Flask development server. To compile a production bundle:

```powershell
npm run build
```

## API

- `GET /api/health`
- `GET /api/scenes`
- `GET /api/data-status` — resolved paths, required COLMAP file presence, reference count, and query-image availability
- `GET /api/scenes/{scene_id}/model`
- `GET /api/scenes/{scene_id}/viewer-camera` — COLMAP-derived initial camera position and direction for the 3DGS viewer
- `POST /api/localize` — multipart fields `image`, `scene_id`
- `POST /api/localize/configured-query` — JSON `scene_id`, using the actual configured `query.jpg` without copying it
- `GET /api/localizations?limit=100`
- `GET /api/localizations/{id}`
- `GET /api/localizations/{id}/visualization`
- `POST /api/copilot/chat` — JSON messages, or multipart messages plus optional `image` and `scene_id`

## Tests

```powershell
cd backend
python -m unittest discover -s tests -v
cd ..\frontend
npm run build
```

Tests use a temporary SQLite database and a deterministic fake localization engine only to verify API persistence/serialization. They do not claim scientific localization accuracy. Actual localization requires the configured COLMAP model, reference images, OpenCV, pycolmap, and a calibrated query camera.

## Reproduce the localization diagnostic

With the configured scene data in place, start Flask and ensure the real `query.jpg` localization result is present in its history. Then, from `backend/`, run:

```powershell
python tools\diagnose_localization.py
```

The script does not change the localization engine or its thresholds. It writes the top-10 retrieval/matching table, a five-image COLMAP same-sequence cross-check, pose/depth/frame checks, and figures 01–07 to `projet/diagnostics/query_localization_20261107/`. The cross-check excludes each query image itself, uses that image's COLMAP intrinsics scaled to processing resolution, and selects among the same top-10 candidates as the engine. Its candidates are adjacent frames from the same reconstruction, so it is an internal consistency check, not an independent generalization benchmark. Scores are labelled HSV fallback unless DINOv2 is installed and enabled. The report also relies on the persisted query result exposed by the running API; it errors instead of fabricating a pose if no such result exists.
