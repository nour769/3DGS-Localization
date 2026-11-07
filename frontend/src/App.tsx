import {
  Activity,
  ArrowRight,
  Bot,
  Box,
  Check,
  ChevronDown,
  CircleHelp,
  Clock3,
  Database,
  FileImage,
  Gauge,
  ImagePlus,
  LayoutDashboard,
  LoaderCircle,
  MapPin,
  Menu,
  MessageSquareText,
  Navigation,
  Radio,
  ScanSearch,
  Send,
  ShieldAlert,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import {
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
  type ReactNode,
  lazy,
  Suspense,
  useEffect,
  useMemo,
  useState,
} from "react";
import { absoluteModelUrl, api } from "./api";
import type {
  DataStatus,
  Localization,
  LocalizationResult,
  Pose,
  Scene,
} from "./types";

const SceneViewer = lazy(() => import("./SceneViewer"));

type Page = "dashboard" | "localize" | "explorer" | "history" | "copilot";
type ChatMessage = { role: "user" | "assistant"; content: string };

const navItems: { id: Page; label: string; icon: typeof LayoutDashboard }[] = [
  { id: "dashboard", label: "Overview", icon: LayoutDashboard },
  { id: "localize", label: "Localize image", icon: ScanSearch },
  { id: "explorer", label: "3D explorer", icon: Box },
  { id: "history", label: "Run history", icon: Clock3 },
  { id: "copilot", label: "AI copilot", icon: Bot },
];

function formatNumber(value: unknown, digits = 2) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";
}

function StatusBadge({ status }: { status: string }) {
  const labels: Record<string, string> = {
    localized: "Localized",
    uncertain: "Uncertain",
    failed: "Failed",
    processing: "Processing",
  };
  return (
    <span className={`status-badge status-${status}`}>
      <span className="status-dot" />
      {labels[status] || status}
    </span>
  );
}

function PageTitle({
  eyebrow,
  title,
  subtitle,
  action,
}: {
  eyebrow: string;
  title: string;
  subtitle: string;
  action?: ReactNode;
}) {
  return (
    <div className="page-title-row">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      {action}
    </div>
  );
}

function EmptyState({
  icon: Icon,
  title,
  detail,
  action,
}: {
  icon: typeof Database;
  title: string;
  detail: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <Icon size={20} />
      </div>
      <h3>{title}</h3>
      <p>{detail}</p>
      {action}
    </div>
  );
}

function Dashboard({
  scenes,
  dataStatus,
  localizations,
  health,
  navigate,
}: {
  scenes: Scene[];
  dataStatus: DataStatus | null;
  localizations: Localization[];
  health: string;
  navigate: (page: Page) => void;
}) {
  const successful = localizations.filter((item) => item.status === "localized").length;
  const medianInliers = useMemo(() => {
    const values = localizations
      .map((item) => item.metrics.inliers)
      .filter((value): value is number => typeof value === "number")
      .sort((a, b) => a - b);
    return values.length ? values[Math.floor(values.length / 2)] : null;
  }, [localizations]);
  const recent = localizations.slice(0, 5);
  return (
    <>
      <PageTitle
        eyebrow="Research workspace"
        title="Visual localization"
        subtitle="Estimate camera pose against registered COLMAP scene observations."
        action={
          <button className="button button-primary" onClick={() => navigate("localize")}>
            <ScanSearch size={16} /> New localization
          </button>
        }
      />
      <div className="health-strip">
        <span className={`health-led ${health === "ok" ? "led-online" : ""}`} />
        API {health === "ok" ? "connected" : "unavailable"}
        <span className="strip-divider" />
        <span>{scenes.filter((scene) => scene.ready).length} ready scene(s)</span>
        <span className="strip-spacer" />
        <span className="mono tiny">COLMAP WORLD · PnP-RANSAC</span>
      </div>
      {dataStatus?.scenes.map((data) => (
        <section className="data-check panel" key={data.id}>
          <div className="panel-heading">
            <div>
              <span className="eyebrow">DATA AVAILABILITY · {data.name}</span>
              <h2>{data.ready ? "Localization inputs ready" : "Some data paths are not available"}</h2>
            </div>
            <span className={`scene-state ${data.ready ? "ready" : "not-ready"}`}>
              {data.ready ? "Ready" : "Incomplete"}
            </span>
          </div>
          <div className="data-check-grid">
            <div><span>3D Gaussian scene</span><strong className={data.ply_available ? "check-ok" : "check-missing"}>{data.ply_available ? "Present" : "Missing"}</strong><small className="mono">{data.ply_path || "GAUSSIAN_PLY_PATH not set"}</small></div>
            <div><span>COLMAP core model</span><strong className={["cameras", "images", "points3D"].every((key) => data.colmap_files[key]) ? "check-ok" : "check-missing"}>{["cameras", "images", "points3D"].filter((key) => data.colmap_files[key]).length}/3 files</strong><small className="mono">{data.colmap_model_path || "COLMAP_MODEL_PATH / PROJECT_DATA_ROOT not set"}</small></div>
            <div><span>Reference images</span><strong className={data.reference_image_count ? "check-ok" : "check-missing"}>{data.reference_image_count} detected</strong><small className="mono">{data.reference_images_path || "REFERENCE_IMAGES_DIR / PROJECT_DATA_ROOT not set"}</small></div>
            <div><span>query.jpg</span><strong className={data.query_image_available ? "check-ok" : "check-missing"}>{data.query_image_available ? "Present" : "Missing"}</strong><small className="mono">{data.query_image_path || "QUERY_IMAGE_PATH / PROJECT_DATA_ROOT not set"}</small></div>
            <div><span>Query calibration</span><strong className={data.query_calib_available ? "check-ok" : "check-missing"}>{data.query_calib_available ? "calib.txt present" : "Using configured intrinsics"}</strong><small className="mono">{data.query_calib_path || "QUERY_CALIB_PATH / DATA_ROOT not set"}</small></div>
            <div><span>COLMAP rigs / frames</span><strong className="check-neutral">{data.colmap_files.rigs ? "rigs present" : "rigs missing"} · {data.colmap_files.frames ? "frames present" : "frames missing"}</strong><small>Core localization needs cameras, images and points3D.</small></div>
            <div><span>PLY / COLMAP frame alignment</span><strong className={data.coordinate_frame_verified ? "check-ok" : "check-missing"}>{data.coordinate_frame_verified ? "Numerically checked" : "Not verified"}</strong><small>{data.coordinate_frame_verified ? "Camera overlays are enabled; localization confidence is reported separately for each pose." : "Do not overlay a pose on the PLY until the reference-view check passes."}</small></div>
          </div>
        </section>
      ))}
      <section className="stat-grid">
        <article className="stat-card">
          <div className="stat-icon cyan"><Box size={17} /></div>
          <span className="stat-label">Configured scenes</span>
          <strong>{scenes.length}</strong>
          <small>{scenes.filter((scene) => scene.ready).length} ready for localization</small>
        </article>
        <article className="stat-card">
          <div className="stat-icon blue"><Activity size={17} /></div>
          <span className="stat-label">Localization runs</span>
          <strong>{localizations.length}</strong>
          <small>Persisted in the database</small>
        </article>
        <article className="stat-card">
          <div className="stat-icon green"><Check size={17} /></div>
          <span className="stat-label">Accepted runs</span>
          <strong>{successful}</strong>
          <small>Current stored results only</small>
        </article>
        <article className="stat-card">
          <div className="stat-icon amber"><Gauge size={17} /></div>
          <span className="stat-label">Median inliers</span>
          <strong>{medianInliers ?? "—"}</strong>
          <small>Across runs with reported metrics</small>
        </article>
      </section>
      <section className="content-grid dashboard-grid">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">DATASETS</span>
              <h2>Scene library</h2>
            </div>
            <button className="button button-quiet" onClick={() => navigate("explorer")}>
              Open explorer <ArrowRight size={14} />
            </button>
          </div>
          {scenes.length ? (
            <div className="scene-list">
              {scenes.slice(0, 4).map((scene) => (
                <div className="scene-row" key={scene.id}>
                  <div className="scene-symbol"><Box size={17} /></div>
                  <div className="scene-main">
                    <strong>{scene.name}</strong>
                    <span>
                      {scene.has_gaussian_ply ? "Gaussian PLY" : "No PLY model"}
                      {" · "}
                      {scene.coordinate_frame_verified
                        ? "Frame verified"
                        : "Frame alignment unverified"}
                    </span>
                  </div>
                  <span className={`scene-state ${scene.ready ? "ready" : "not-ready"}`}>
                    {scene.ready ? "Ready" : "Setup required"}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <EmptyState
              icon={Database}
              title="No scene data configured"
              detail="Point SCENES_DIR to a scene folder containing a COLMAP sparse model and reference images."
            />
          )}
        </div>
        <div className="panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">RECENT ACTIVITY</span>
              <h2>Latest runs</h2>
            </div>
            <button className="icon-button" aria-label="View history" onClick={() => navigate("history")}>
              <ArrowRight size={16} />
            </button>
          </div>
          {recent.length ? (
            <div className="activity-list">
              {recent.map((item) => (
                <div className="activity-row" key={item.id}>
                  <div className={`activity-icon activity-${item.status}`}>
                    {item.status === "localized" ? <Check size={15} /> : <MapPin size={15} />}
                  </div>
                  <div className="activity-main">
                    <strong>{item.image_name}</strong>
                    <span>{item.scene_id} · {item.created_at ? new Date(item.created_at).toLocaleString() : "—"}</span>
                  </div>
                  <StatusBadge status={item.status} />
                </div>
              ))}
            </div>
          ) : (
            <EmptyState
              icon={Activity}
              title="No activity yet"
              detail="Completed localization attempts will appear here."
              action={<button className="text-button" onClick={() => navigate("localize")}>Start a run <ArrowRight size={13} /></button>}
            />
          )}
        </div>
      </section>
      <div className="notice notice-subtle">
        <CircleHelp size={15} />
        <span>
          Preliminary notebook metrics are not platform benchmarks. Translation is reported in
          COLMAP scene units—not metres unless independently verified.
        </span>
      </div>
    </>
  );
}

function LocalizationPage({
  scenes,
  queryImageAvailable,
  queryImagePath,
  onComplete,
}: {
  scenes: Scene[];
  queryImageAvailable: boolean;
  queryImagePath: string | null;
  onComplete: (result: LocalizationResult) => void;
}) {
  const readyScenes = scenes.filter((scene) => scene.ready);
  const [sceneId, setSceneId] = useState(readyScenes[0]?.id || "");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [queryBusy, setQueryBusy] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const chooseFile = (selected: File | null) => {
    if (!selected) return;
    setError("");
    if (!selected.type.startsWith("image/")) {
      setError("Choose a supported image file.");
      return;
    }
    if (selected.size > 20 * 1024 * 1024) {
      setError("Image must be smaller than 20 MB.");
      return;
    }
    setFile(selected);
  };

  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    chooseFile(event.dataTransfer.files[0] || null);
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!file || !sceneId) return;
    setBusy(true);
    setError("");
    try {
      const result = await api.localize(file, sceneId);
      onComplete(result);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Localization request failed.");
    } finally {
      setBusy(false);
    }
  };

  const submitConfiguredQuery = async () => {
    if (!sceneId) return;
    setQueryBusy(true);
    setError("");
    try {
      onComplete(await api.localizeConfiguredQuery(sceneId));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Configured query localization failed.");
    } finally {
      setQueryBusy(false);
    }
  };

  return (
    <>
      <PageTitle
        eyebrow="POSE ESTIMATION"
        title="Localize an image"
        subtitle="Upload a query frame and select its registered reference scene."
      />
      <form className="localize-layout" onSubmit={submit}>
        <div className="panel upload-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">QUERY IMAGE</span><h2>Image input</h2></div>
            <span className="step-count">01 / 02</span>
          </div>
          <label
            className={`drop-zone ${dragging ? "drop-active" : ""} ${preview ? "has-preview" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp,image/bmp"
              onChange={(event: ChangeEvent<HTMLInputElement>) => chooseFile(event.target.files?.[0] || null)}
            />
            {preview ? (
              <>
                <img src={preview} alt="Selected query frame" className="query-preview" />
                <span className="preview-chip"><FileImage size={13} /> {file?.name}</span>
                <button type="button" className="remove-file" aria-label="Remove image" onClick={(event) => { event.preventDefault(); setFile(null); }}>
                  <X size={15} />
                </button>
              </>
            ) : (
              <div className="upload-prompt">
                <div className="upload-icon"><Upload size={19} /></div>
                <strong>Drop a road image here</strong>
                <span>or <u>browse files</u> · JPG, PNG, WebP · up to 20 MB</span>
              </div>
            )}
          </label>
          <div className="field-note"><ShieldAlert size={14} /> Images are stored on this server with localization history.</div>
        </div>
        <div className="panel run-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">RECONSTRUCTION</span><h2>Scene selection</h2></div>
            <span className="step-count">02 / 02</span>
          </div>
          <label className="field-label" htmlFor="scene-select">Reference scene</label>
          <div className="select-wrap">
            <select id="scene-select" value={sceneId} onChange={(event) => setSceneId(event.target.value)} disabled={!readyScenes.length}>
              {!readyScenes.length && <option value="">No ready scenes configured</option>}
              {readyScenes.map((scene) => <option value={scene.id} key={scene.id}>{scene.name}</option>)}
            </select>
            <ChevronDown size={16} />
          </div>
          <div className="scene-facts">
            <div><span>Pose solver</span><strong>SIFT + PnP-RANSAC</strong></div>
            <div><span>World coordinates</span><strong>COLMAP · scene units</strong></div>
            <div><span>Camera calibration</span><strong>Scaled query intrinsics</strong></div>
          </div>
          <div className="method-note">
            <Sparkles size={15} />
            <p>2D features are matched to observed COLMAP 3D points. This avoids inferring 3D points from Gaussian depth.</p>
          </div>
          {error && <div className="form-error">{error}</div>}
          <button className="button button-primary run-button" disabled={!file || !sceneId || busy}>
            {busy ? <><LoaderCircle className="spin" size={17} /> Estimating pose…</> : <><ScanSearch size={17} /> Run localization</>}
          </button>
          <button
            type="button"
            className="button button-secondary run-button"
            disabled={!queryImageAvailable || !sceneId || queryBusy || busy}
            onClick={() => void submitConfiguredQuery()}
          >
            {queryBusy ? <><LoaderCircle className="spin" size={16} /> Testing query.jpg…</> : <><FileImage size={16} /> Localize configured query.jpg</>}
          </button>
          {!queryImageAvailable && (
            <div className="query-path-note">
              <ShieldAlert size={13} />
              <span>query.jpg not accessible{queryImagePath ? ` at ${queryImagePath}` : ". Set PROJECT_DATA_ROOT or QUERY_IMAGE_PATH."}</span>
            </div>
          )}
        </div>
      </form>
    </>
  );
}

function ResultPanel({
  item,
  scene,
  onExplore,
}: {
  item: Localization;
  scene?: Scene;
  onExplore: (item: Localization) => void;
}) {
  const pose = item.pose;
  return (
    <div className={`result-panel result-${item.status}`}>
      <div className="result-heading">
        <div className="result-icon">{item.status === "localized" ? <Check size={18} /> : <MapPin size={18} />}</div>
        <div>
          <div className="result-title">{item.status === "localized" ? "Pose estimated" : item.status === "uncertain" ? "Pose needs review" : "Localization failed"}</div>
          <div className="result-subtitle">{item.error || (item.status === "localized" ? "Confidence thresholds met for this run." : "Review metrics before using this estimate.")}</div>
        </div>
        <StatusBadge status={item.status} />
      </div>
      {pose ? (
        <>
          {typeof item.metrics.calibration_warning === "string" && item.metrics.calibration_warning && (
            <div className="notice notice-warning calibration-notice"><ShieldAlert size={14} /><span>{item.metrics.calibration_warning}</span></div>
          )}
          <div className="result-metrics">
            <div><span>INLIERS</span><strong>{item.metrics.inliers ?? "—"}</strong><small>{typeof item.metrics.inlier_ratio === "number" ? `${(item.metrics.inlier_ratio * 100).toFixed(1)}% of matches` : "—"}</small></div>
            <div><span>MEDIAN REPROJECTION</span><strong>{formatNumber(item.metrics.median_reprojection_error_px)} <i>px</i></strong><small>on PnP inliers</small></div>
            <div><span>REFERENCE</span><strong className="reference-name">{item.metrics.reference_name || "—"}</strong><small>{item.metrics.retrieval_method || "retrieval unavailable"}</small></div>
          </div>
          <div className="pose-grid">
            <div className="pose-block"><span><Navigation size={13} /> CAMERA CENTER</span>
              <strong>{pose.camera_center.map((value) => formatNumber(value, 3)).join(" , ")}</strong>
              <small>X · Y · Z in COLMAP scene units</small>
            </div>
            <div className="pose-block"><span><Radio size={13} /> ORIENTATION</span>
              <strong>{pose.euler_xyz_degrees.map((value) => `${formatNumber(value, 1)}°`).join(" , ")}</strong>
              <small>Euler XYZ · quaternion wxyz</small>
            </div>
          </div>
          <div className="result-footer">
            <span>{scene?.coordinate_frame_verified ? "COLMAP / Gaussian frame verified" : "Gaussian / COLMAP alignment not verified"}</span>
            <button className="button button-secondary" onClick={() => onExplore(item)}>Open in 3D explorer <ArrowRight size={14} /></button>
          </div>
        </>
      ) : (
        <div className="failure-detail">
          <strong>No camera pose was returned.</strong>
          <span>{item.metrics.retrieval_method ? `Retrieval: ${item.metrics.retrieval_method}` : "Check the scene model, camera calibration, and feature overlap."}</span>
        </div>
      )}
    </div>
  );
}

function HistoryPage({
  localizations,
  scenes,
  onExplore,
}: {
  localizations: Localization[];
  scenes: Scene[];
  onExplore: (item: Localization) => void;
}) {
  return (
    <>
      <PageTitle eyebrow="DATABASE" title="Localization history" subtitle="Browse persisted attempts, metrics, and estimated camera poses." />
      <div className="panel history-panel">
        <div className="table-heading"><div><h2>All runs</h2><span>{localizations.length} records</span></div><div className="table-caption"><Database size={14} /> PostgreSQL / SQLAlchemy</div></div>
        {localizations.length ? (
          <div className="table-scroll">
            <table>
              <thead><tr><th>QUERY IMAGE</th><th>SCENE</th><th>STATUS</th><th>INLIERS</th><th>MEDIAN ERROR</th><th>DATE</th><th /></tr></thead>
              <tbody>
                {localizations.map((item) => (
                  <tr key={item.id}>
                    <td><div className="image-cell"><div className="mini-image"><FileImage size={14} /></div><div><strong>{item.image_name}</strong><small className="mono">{item.id.slice(0, 8)}</small></div></div></td>
                    <td>{scenes.find((scene) => scene.id === item.scene_id)?.name || item.scene_id}</td>
                    <td><StatusBadge status={item.status} /></td>
                    <td className="mono">{item.metrics.inliers ?? "—"}</td>
                    <td className="mono">{typeof item.metrics.median_reprojection_error_px === "number" ? `${formatNumber(item.metrics.median_reprojection_error_px)} px` : "—"}</td>
                    <td className="table-date">{item.created_at ? new Date(item.created_at).toLocaleString() : "—"}</td>
                    <td><button className="icon-button" disabled={!item.pose} aria-label="Open camera in 3D explorer" onClick={() => onExplore(item)}><ArrowRight size={15} /></button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState icon={Clock3} title="No saved localizations" detail="Start a localization run to add the first record." />
        )}
      </div>
    </>
  );
}

function ExplorerPage({
  scenes,
  localizations,
  selected,
}: {
  scenes: Scene[];
  localizations: Localization[];
  selected: Localization | null;
}) {
  const [sceneId, setSceneId] = useState(selected?.scene_id || scenes[0]?.id || "");
  const [selectedId, setSelectedId] = useState(selected?.id || "");
  useEffect(() => {
    if (!sceneId && scenes.length) setSceneId(scenes[0].id);
  }, [sceneId, scenes]);
  useEffect(() => {
    if (selected) {
      setSelectedId(selected.id);
      setSceneId(selected.scene_id);
    }
  }, [selected]);
  const scene = scenes.find((item) => item.id === sceneId);
  const sceneResults = useMemo(
    () => localizations.filter((item) => item.scene_id === sceneId && item.pose),
    [localizations, sceneId],
  );
  const active = localizations.find((item) => item.id === selectedId) || sceneResults[0] || null;
  const visiblePoses = useMemo(
    () => sceneResults.slice(0, 8).map((item) => item.pose as Pose),
    [sceneResults],
  );
  const modelUrl = scene?.model_url ? absoluteModelUrl(scene.model_url) : null;
  return (
    <>
      <PageTitle eyebrow="SPATIAL VIEW" title="3D scene explorer" subtitle="Inspect the photorealistic Gaussian scene and estimated camera poses." />
      {!scenes.length ? (
        <div className="panel"><EmptyState icon={Box} title="No scenes configured" detail="Configure SCENES_DIR and restart the backend to load a COLMAP model." /></div>
      ) : (
        <>
          <div className="explorer-toolbar panel">
            <label><span>SCENE</span><select value={sceneId} onChange={(event) => setSceneId(event.target.value)}>{scenes.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
            <label><span>CAMERA POSE</span><select value={active?.id || ""} onChange={(event) => setSelectedId(event.target.value)}><option value="">No pose selected</option>{sceneResults.map((item) => <option value={item.id} key={item.id}>{item.image_name} · {item.status}</option>)}</select></label>
            <div className="toolbar-info"><span className="health-led led-online" /> {visiblePoses.length} pose(s) available</div>
          </div>
          {scene?.has_gaussian_ply && !scene.coordinate_frame_verified && (
            <div className="notice notice-warning"><ShieldAlert size={16} /><span>Gaussian PLY / COLMAP alignment has not been verified. Camera overlays are disabled; validate the frame in a known reference view before enabling <span className="mono">SCENE_FRAME_VERIFIED</span>.</span></div>
          )}
          <div className="explorer-layout">
            <div className="panel viewer-panel">
              <div className="viewer-heading"><div><span className="eyebrow">SCENE GEOMETRY</span><h2>{scene?.name}</h2></div><div className="viewer-tools"><span><span className="legend-dot cloud" /> Gaussian splats</span><span><span className="legend-dot camera" /> Camera pose</span></div></div>
              <Suspense fallback={<div className="viewer-shell viewer-loading">Loading 3D viewer…</div>}>
                <SceneViewer modelUrl={modelUrl} sceneId={scene?.id ?? null} poses={visiblePoses} overlayAllowed={Boolean(scene?.coordinate_frame_verified)} />
              </Suspense>
              {!scene?.has_gaussian_ply && <div className="viewer-footnote">No Gaussian PLY configured. Showing the COLMAP coordinate frame and available camera poses only.</div>}
            </div>
            <aside className="panel pose-sidebar">
              <div className="panel-heading"><div><span className="eyebrow">SELECTED RESULT</span><h2>Camera pose</h2></div></div>
              {active?.pose ? (
                <div className="pose-details">
                  <StatusBadge status={active.status} />
                  <div className="pose-detail-image"><ImagePlus size={16} /><span>{active.image_name}</span></div>
                  <div className="detail-group"><span>CAMERA CENTER · COLMAP UNITS</span><strong>{active.pose.camera_center.map((value) => formatNumber(value, 4)).join("\n")}</strong></div>
                  <div className="detail-group"><span>ORIENTATION · EULER XYZ</span><strong>{active.pose.euler_xyz_degrees.map((value) => `${formatNumber(value, 2)}°`).join(" · ")}</strong></div>
                  <div className="detail-group"><span>INLIERS / MEDIAN ERROR</span><strong>{active.metrics.inliers ?? "—"} / {formatNumber(active.metrics.median_reprojection_error_px)} px</strong></div>
                  <div className="detail-group"><span>POSE CONVENTION</span><strong className="small-strong">World → camera (COLMAP)</strong></div>
                </div>
              ) : (
                <EmptyState icon={MapPin} title="No camera pose" detail="Run localization or select a result from history." />
              )}
            </aside>
          </div>
        </>
      )}
    </>
  );
}

function CopilotPage({ scenes }: { scenes: Scene[] }) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content: "I can explain the localization pipeline, inspect stored runs, and help interpret pose metrics. I only report values retrieved from this workspace.",
    },
  ]);
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [sceneId, setSceneId] = useState(scenes.find((scene) => scene.ready)?.id || "");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");

  const send = async (event: FormEvent) => {
    event.preventDefault();
    if ((!text.trim() && !file) || sending) return;
    const content = text.trim() || "Please localize the attached image.";
    const next = [...messages, { role: "user" as const, content }];
    setMessages(next);
    setText("");
    setSending(true);
    setError("");
    try {
      const result = await api.copilot(next, file, sceneId);
      setMessages([...next, { role: "assistant", content: result.reply }]);
      setFile(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Copilot request failed.");
      setMessages(next);
    } finally {
      setSending(false);
    }
  };

  return (
    <>
      <PageTitle eyebrow="RESEARCH ASSISTANT" title="AI copilot" subtitle="Ask about actual scenes, stored results, and the localization method." action={<div className="ai-ready"><span className="health-led" /> Provider configured by backend</div>} />
      <div className="copilot-layout">
        <section className="panel chat-panel">
          <div className="chat-heading"><div className="copilot-avatar"><Bot size={18} /></div><div><strong>Fieldnotes Copilot</strong><span>Grounded in workspace data · no fabricated results</span></div><span className="online-label"><span className="health-led led-online" /> READY</span></div>
          <div className="chat-messages">
            {messages.map((message, index) => (
              <div className={`chat-message message-${message.role}`} key={`${index}-${message.role}`}>
                <div className="message-avatar">{message.role === "assistant" ? <Sparkles size={14} /> : "Y"}</div>
                <div className="message-body"><span>{message.role === "assistant" ? "COPILOT" : "YOU"}</span><p>{message.content}</p></div>
              </div>
            ))}
            {sending && <div className="typing-state"><LoaderCircle className="spin" size={15} /> Retrieving workspace data…</div>}
          </div>
          <form className="chat-composer" onSubmit={send}>
            {file && <div className="attachment-chip"><FileImage size={13} /> {file.name}<button type="button" aria-label="Remove attachment" onClick={() => setFile(null)}><X size={13} /></button></div>}
            <div className="composer-row">
              <label className="attach-button" title="Attach image"><input type="file" accept="image/*" onChange={(event) => setFile(event.target.files?.[0] || null)} /><ImagePlus size={17} /></label>
              <input value={text} onChange={(event) => setText(event.target.value)} placeholder="Ask about a run, pose, or localization method…" />
              <button className="send-button" disabled={sending || (!text.trim() && !file)} aria-label="Send message"><Send size={16} /></button>
            </div>
            {file && (
              <div className="composer-scene">
                <span>Run against</span>
                <select value={sceneId} onChange={(event) => setSceneId(event.target.value)}>
                  <option value="">Choose a scene</option>
                  {scenes.filter((scene) => scene.ready).map((scene) => <option key={scene.id} value={scene.id}>{scene.name}</option>)}
                </select>
                <small>Explicitly ask to localize the attached image to run it.</small>
              </div>
            )}
            {error && <div className="form-error">{error}</div>}
            <div className="composer-footnote"><ShieldAlert size={12} /> Configure LLM_API_KEY for generated responses. No key? API data remains accessible.</div>
          </form>
        </section>
        <aside className="copilot-aside">
          <div className="panel copilot-info">
            <div className="copilot-aside-icon"><MessageSquareText size={17} /></div>
            <h3>What I can help with</h3>
            <ul><li>Explain inliers and reprojection error.</li><li>Retrieve and compare saved localizations.</li><li>Describe scene readiness and pipeline limits.</li><li>Run localization after an explicit request and image upload.</li></ul>
          </div>
          <div className="notice notice-subtle"><ShieldAlert size={15} /><span>Requests are sent only to the API provider configured by the operator. Avoid uploading sensitive imagery.</span></div>
        </aside>
      </div>
    </>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("dashboard");
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [dataStatus, setDataStatus] = useState<DataStatus | null>(null);
  const [localizations, setLocalizations] = useState<Localization[]>([]);
  const [health, setHealth] = useState("checking");
  const [selectedLocalization, setSelectedLocalization] = useState<Localization | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [latestResult, setLatestResult] = useState<Localization | null>(null);
  const [toastVisible, setToastVisible] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  const refresh = async () => {
    setLoading(true);
    const [healthResult2, sceneResult2, dataStatusResult, localizationResult] = await Promise.allSettled([
      api.health(),
      api.scenes(),
      api.dataStatus(),
      api.localizations(),
    ]);
    setHealth(healthResult2.status === "fulfilled" ? healthResult2.value.status : "unavailable");
    if (sceneResult2.status === "fulfilled") setScenes(sceneResult2.value);
    if (dataStatusResult.status === "fulfilled") setDataStatus(dataStatusResult.value);
    if (localizationResult.status === "fulfilled") setLocalizations(localizationResult.value);
    const failed = [sceneResult2, dataStatusResult, localizationResult].find((item) => item.status === "rejected");
    if (failed?.status === "rejected") {
      setLoadError(failed.reason instanceof Error ? failed.reason.message : "Could not load workspace data.");
    } else {
      setLoadError("");
    }
    setLoading(false);
  };

  useEffect(() => { void refresh(); }, []);

  const handleComplete = (response: LocalizationResult) => {
    const item = response.localization;
    setLatestResult(item);
    setToastVisible(true);
    window.setTimeout(() => setToastVisible(false), 4500);
    setLocalizations((current) => [item, ...current.filter((entry) => entry.id !== item.id)]);
  };

  const explore = (item: Localization) => {
    setSelectedLocalization(item);
    setPage("explorer");
  };

  const nav = (next: Page) => {
    setPage(next);
    setMobileNavOpen(false);
  };

  const currentLabel = navItems.find((item) => item.id === page)?.label || "Overview";
  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNavOpen ? "sidebar-open" : ""}`}>
        <div className="brand"><div className="brand-mark"><Navigation size={18} /></div><div><strong>FIELDNOTES</strong><span>VISION PLATFORM</span></div></div>
        <div className="workspace-switch"><div className="workspace-dot">R</div><div><span>WORKSPACE</span><strong>Road scene research</strong></div><ChevronDown size={14} /></div>
        <div className="nav-section-label">WORKSPACE</div>
        <nav className="main-nav">
          {navItems.map(({ id, label, icon: Icon }) => (
            <button className={`nav-item ${page === id ? "nav-active" : ""}`} key={id} onClick={() => nav(id)}>
              <Icon size={17} strokeWidth={1.8} /><span>{label}</span>
              {id === "copilot" && <span className="nav-new"><Sparkles size={11} /></span>}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="backend-card"><div className="backend-card-title"><span className={`health-led ${health === "ok" ? "led-online" : ""}`} /> Backend status</div><strong>{health === "ok" ? "Connected" : health === "checking" ? "Connecting…" : "Offline"}</strong><span>Flask API · /api/health</span></div>
          <div className="sidebar-footer"><span>3DGS LOCALIZATION</span><span>v0.1.0 · research build</span></div>
        </div>
      </aside>
      {mobileNavOpen && <button className="mobile-backdrop" aria-label="Close navigation" onClick={() => setMobileNavOpen(false)} />}
      <main className="main-shell">
        <header className="topbar">
          <button className="mobile-menu" aria-label="Open navigation" onClick={() => setMobileNavOpen(true)}><Menu size={19} /></button>
          <div className="breadcrumbs"><span>Workspace</span><span className="breadcrumb-sep">/</span><strong>{currentLabel}</strong></div>
          <div className="topbar-right"><span className="live-tag"><span className="health-led led-online" /> LOCAL ENVIRONMENT</span><span className="topbar-divider" /><div className="avatar">RM</div></div>
        </header>
        <div className="page-content">
          {loadError && <div className="notice notice-error"><ShieldAlert size={16} /><span>Backend data could not be fully loaded: {loadError}</span><button className="text-button" onClick={() => void refresh()}>Retry</button></div>}
          {latestResult && page === "localize" && (
            <div className="localization-result-wrap">
              <div className="result-wrap-heading"><span className="eyebrow">LATEST ATTEMPT</span><button className="icon-button" aria-label="Dismiss latest result" onClick={() => setLatestResult(null)}><X size={15} /></button></div>
              <ResultPanel item={latestResult} scene={scenes.find((scene) => scene.id === latestResult.scene_id)} onExplore={explore} />
            </div>
          )}
          {loading && page === "dashboard" ? <div className="loading-view"><LoaderCircle className="spin" size={19} /> Connecting to localization services…</div> : null}
          {page === "dashboard" && !loading && <Dashboard scenes={scenes} dataStatus={dataStatus} localizations={localizations} health={health} navigate={nav} />}
          {page === "localize" && (
            <LocalizationPage
              scenes={scenes}
              queryImageAvailable={Boolean(dataStatus?.scenes.some((scene) => scene.query_image_available))}
              queryImagePath={dataStatus?.scenes.find((scene) => scene.query_image_path)?.query_image_path || null}
              onComplete={handleComplete}
            />
          )}
          {page === "history" && <HistoryPage scenes={scenes} localizations={localizations} onExplore={explore} />}
          {page === "explorer" && <ExplorerPage scenes={scenes} localizations={localizations} selected={selectedLocalization} />}
          {page === "copilot" && <CopilotPage scenes={scenes} />}
          <footer className="page-footer"><span>Fieldnotes <span className="footer-dot">·</span> 3DGS Visual Localization</span><span>Scientific outputs are experimental until validated against held-out references.</span></footer>
        </div>
      </main>
      {toastVisible && latestResult && page === "localize" && (
        <div className="toast"><Check size={15} /> Localization attempt saved <button aria-label="Dismiss notification" onClick={() => setToastVisible(false)}><X size={14} /></button></div>
      )}
    </div>
  );
}
