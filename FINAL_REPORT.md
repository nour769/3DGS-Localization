# Final report — 3DGS Visual Localization with AI Copilot

## Implemented

- Flask REST backend with SQLAlchemy persistence, scene discovery, validated image uploads, localization history/detail/visualization APIs, and health checks.
- Modular scientific engine: COLMAP observations → SIFT feature matches → OpenCV PnP-RANSAC/refinement. Uses the configured query intrinsics at their calibration resolution and scales them to uploaded/working dimensions. Pose output includes world-to-camera transform, camera center, quaternion, Euler XYZ, inliers, inlier ratio, and reprojection metrics.
- Optional DINOv2 retrieval and an explicitly reported color-histogram retrieval fallback.
- React/TypeScript/Vite/Tailwind dashboard, upload/localization workflow, status and metric results, history, Copilot chat, and the official SuperSplat/PlayCanvas Gaussian Splatting viewer.
- PLY camera overlays are gated by `SCENE_FRAME_VERIFIED`; translation is labelled in COLMAP scene units, not metres.
- OpenAI-compatible Copilot with retrieval/localization tool calls, explicit-request/image checks, and a no-key fallback.
- Environment examples, setup documentation, and backend API/geometry tests.

## Project structure

- `backend/app/localization/`: camera geometry, COLMAP loading, and localization engine.
- `backend/app/routes/`: Flask API endpoints.
- `backend/app/services/`: scene discovery, localization persistence, and Copilot integration.
- `backend/tests/`: geometry and REST API tests.
- `frontend/src/`: app pages, API client, types, and the SuperSplat/PlayCanvas viewer.

## Verification

- Backend: `python -m unittest discover -s tests -v` — **24 tests passed**, including input-quality checks and a real-engine API failure path.
- Frontend: `npm run build` — **passed** (`tsc -b` and Vite). Vite warns that the lazy SuperSplat chunk is 2.4 MB minified (649 KB gzip); main app bundle is 185 KB (58 KB gzip).
- Moved the three required COLMAP binaries, the 447 reference images, `query.jpg`, and `calib.txt` from `rendu/` to the configured external `projet\data` tree. PyCOLMAP reads the model (447 registered images, 99,150 points); all registered image paths exist.
- Live `GET /api/data-status` returns HTTP 200 and reports the road scene ready, all three core COLMAP files present, 447 references, and the query and calibration available.
- The real `query.jpg` run estimated a pose but returned **uncertain**: 6 PnP inliers from 23 correspondences (26.1%); this is below the configured 15-inlier acceptance threshold. The pose is not an accepted localization, and the near-zero reprojection error from six points is not evidence of accuracy.
- Localization no longer attempts matching on nearly black or uniform images. The actual configured `query.jpg` is almost entirely black, so the current engine rejects it with a clear message before loading the scene index; the acceptance thresholds for real images are unchanged. A genuine road query is still needed to evaluate a new localization.
- The main interface now presents the estimated position, orientation and a plain-language reliability status. Algorithm names and raw data paths are removed from the normal workflow; pixel-level diagnostics are tucked under “Détails supplémentaires”. The scene selection also initializes when scene data arrives asynchronously.
- A reproducible diagnostic is available at `backend/tools/diagnose_localization.py`; its detailed French report, CSV tables, and seven figures are generated under `projet/diagnostics/query_localization_20261107/`. It confirms `query.jpg` is a black test card rather than a road image, and the current retrieval scores are HSV fallback scores (DINOv2/PyTorch are not active). The diagnostic does not change localization thresholds or engine behavior.
- Five same-reconstruction reference-image checks localized adjacent frames with 484–1,744 inliers and camera-center error below 0.001 COLMAP units using each query's own scaled COLMAP calibration. These are correlated neighboring views from the training reconstruction, not an independent validation set or a performance claim for new query photos.
- A point-cloud overlap check found 99.94% of COLMAP points inside the PLY bounds; nearest PLY-center distances are ≤0.05 scene units for 94.59% and ≤0.5 units for 99.73%. This supports a shared coordinate frame, but does not verify metric scale or one-to-one correspondences.
- The browser loaded the configured 471,603-splat scene with SH degree 3 and displayed the existing two uncertain query records. The scene uses the official SuperSplat Gaussian renderer, not Three.js points.
- Fixed the explorer's initial scene selection when scene data arrives asynchronously, and changed startup framing to use the registered COLMAP camera nearest the trajectory median, with that camera's calibrated vertical FOV. `GET /api/scenes/scene-routiere-finale-2/viewer-camera` returned HTTP 200 with 447 registered cameras, camera center `[0.7575, 0.1932, -1.3230]`, and FOV `27.81°`.
- The browser capture still does not establish visual parity with SuperSplat's manually adjusted view; the camera-based first view appears dark/cropped compared with reference image `0000000140.png`. The PLY loads and is rendered as Gaussians, but the initial visual framing/axis orientation needs further investigation. No rendering quality or localization-accuracy claim is made from that capture.
- API integration tests use an in-memory SQLite database and a test-only fake pose engine; they validate transport/persistence behavior, not pose accuracy.

## External data and credentials

- The notebook was inspected and left unchanged. The road Gaussian PLY is present in the project directory and remains unmodified. The imported COLMAP and query assets now live under the external `projet\data` folder; none were added to Git.
- `backend/.env` now points to the supplied road PLY so it appears in the 3D explorer. The adjacent `tandt_db.zip` contains four separate COLMAP datasets (train, truck, drjohnson, playroom), not the road scene; those reconstructions are not falsely attached to the road PLY.
- The road PLY scene is configured and the API serves its 116,959,075-byte model. The Flask API and Vite frontend were started locally at `http://127.0.0.1:5000` and `http://127.0.0.1:5173`.
- The notebook's cached Colab output records 447 cameras/images, 99,150 COLMAP 3D points, and 447 indexed references. Its reference-view render and the local point-cloud overlap check support frame compatibility; no metric-scale verification is claimed.
- Added an external `DATA_ROOT` (`..\..\data` by default), individual path overrides, `calib.txt` loading/scaling, `GET /api/data-status`, and `POST /api/localize/configured-query`. Data-readiness details remain available through the API for operators; the everyday interface focuses on scene availability and localization results.
- The README gives browser-only Google Drive download and extraction instructions; Google Drive Desktop is not required. The camera overlay is enabled for this scene after the measured overlap check. Its current query pose remains marked uncertain and must not be interpreted as a reliable localization.
- No scale-in-metres verification or validated query localization accuracy is claimed.
- DINOv2 weights and an LLM API credential are optional external requirements. SQLite is the local fallback; PostgreSQL is configurable.

## Launch commands (PowerShell)

Backend:

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
python run.py
```

Frontend, in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Run checks:

```powershell
cd backend
python -m unittest discover -s tests -v
cd ..\frontend
npm run build
```
