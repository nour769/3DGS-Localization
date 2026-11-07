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
import type { Localization, LocalizationResult, Pose, Scene } from "./types";

const SceneViewer = lazy(() => import("./SceneViewer"));

type Page = "dashboard" | "localize" | "explorer" | "history" | "copilot";
type ChatMessage = { role: "user" | "assistant"; content: string };

const navItems: { id: Page; label: string; icon: typeof LayoutDashboard }[] = [
  { id: "dashboard", label: "Accueil", icon: LayoutDashboard },
  { id: "localize", label: "Trouver une position", icon: ScanSearch },
  { id: "explorer", label: "Scène 3D", icon: Box },
  { id: "history", label: "Historique", icon: Clock3 },
  { id: "copilot", label: "Assistant", icon: Bot },
];

function formatNumber(value: unknown, digits = 2) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";
}

function StatusBadge({ status }: { status: string }) {
  const labels: Record<string, string> = {
    localized: "Position trouvée",
    uncertain: "À vérifier",
    failed: "Échec",
    processing: "En cours",
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
  localizations,
  health,
  navigate,
}: {
  scenes: Scene[];
  localizations: Localization[];
  health: string;
  navigate: (page: Page) => void;
}) {
  const successful = localizations.filter((item) => item.status === "localized").length;
  const needsReview = localizations.filter((item) => item.status === "uncertain").length;
  const readyScenes = scenes.filter((scene) => scene.ready).length;
  const recent = localizations.slice(0, 5);
  return (
    <>
      <PageTitle
        eyebrow="ESPACE DE TRAVAIL"
        title="Position d'une image"
        subtitle="Importez une photo pour estimer où elle a été prise dans la scène."
        action={
          <button className="button button-primary" onClick={() => navigate("localize")}>
            <ScanSearch size={16} /> Trouver une position
          </button>
        }
      />
      <div className="health-strip">
        <span className={`health-led ${health === "ok" ? "led-online" : ""}`} />
        Service {health === "ok" ? "disponible" : "indisponible"}
        <span className="strip-divider" />
        <span>{readyScenes} scène(s) disponible(s)</span>
      </div>
      <section className="stat-grid">
        <article className="stat-card">
          <div className="stat-icon cyan"><Activity size={17} /></div>
          <span className="stat-label">Photos traitées</span>
          <strong>{localizations.length}</strong>
          <small>Résultats enregistrés</small>
        </article>
        <article className="stat-card">
          <div className="stat-icon green"><Check size={17} /></div>
          <span className="stat-label">Positions trouvées</span>
          <strong>{successful}</strong>
          <small>Assez de détails concordants</small>
        </article>
        <article className="stat-card">
          <div className="stat-icon amber"><CircleHelp size={17} /></div>
          <span className="stat-label">À vérifier</span>
          <strong>{needsReview}</strong>
          <small>Résultats à ne pas utiliser seuls</small>
        </article>
        <article className="stat-card">
          <div className="stat-icon blue"><Box size={17} /></div>
          <span className="stat-label">Scènes disponibles</span>
          <strong>{readyScenes}</strong>
          <small>{readyScenes ? "Prêtes à recevoir une photo" : "Configuration nécessaire"}</small>
        </article>
      </section>
      <section className="content-grid dashboard-grid">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">SCÈNES</span>
              <h2>Scènes disponibles</h2>
            </div>
            <button className="button button-quiet" onClick={() => navigate("explorer")}>
              Voir la scène <ArrowRight size={14} />
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
                      {scene.ready ? "Prête pour estimer une position" : "Configuration nécessaire"}
                    </span>
                  </div>
                  <span className={`scene-state ${scene.ready ? "ready" : "not-ready"}`}>
                    {scene.ready ? "Disponible" : "À configurer"}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <EmptyState
              icon={Database}
              title="Aucune scène disponible"
              detail="Une scène doit être configurée avant de pouvoir estimer une position."
            />
          )}
        </div>
        <div className="panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">RÉSULTATS RÉCENTS</span>
              <h2>Dernières photos</h2>
            </div>
            <button className="icon-button" aria-label="Voir l’historique" onClick={() => navigate("history")}>
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
                    <span>{scenes.find((scene) => scene.id === item.scene_id)?.name || item.scene_id} · {item.created_at ? new Date(item.created_at).toLocaleString("fr-FR") : "—"}</span>
                  </div>
                  <StatusBadge status={item.status} />
                </div>
              ))}
            </div>
          ) : (
            <EmptyState
              icon={Activity}
              title="Aucun résultat pour le moment"
              detail="Les positions estimées apparaîtront ici."
              action={<button className="text-button" onClick={() => navigate("localize")}>Choisir une photo <ArrowRight size={13} /></button>}
            />
          )}
        </div>
      </section>
      <div className="notice notice-subtle">
        <CircleHelp size={15} />
        <span>Les positions indiquées « À vérifier » sont approximatives et ne doivent pas être utilisées seules.</span>
      </div>
    </>
  );
}

function LocalizationPage({
  scenes,
  onComplete,
}: {
  scenes: Scene[];
  onComplete: (result: LocalizationResult) => void;
}) {
  const readyScenes = useMemo(() => scenes.filter((scene) => scene.ready), [scenes]);
  const [sceneId, setSceneId] = useState(readyScenes[0]?.id || "");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  useEffect(() => {
    if (readyScenes.length && !readyScenes.some((scene) => scene.id === sceneId)) {
      setSceneId(readyScenes[0].id);
    }
  }, [readyScenes, sceneId]);

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

  return (
    <>
      <PageTitle
        eyebrow="ESTIMATION DE POSITION"
        title="Où cette photo a-t-elle été prise ?"
        subtitle="Choisissez une photo nette et la scène correspondante. Nous vous montrerons le résultat et son niveau de fiabilité."
      />
      <form className="localize-layout" onSubmit={submit}>
        <div className="panel upload-panel">
          <div className="panel-heading">
          <div><span className="eyebrow">VOTRE PHOTO</span><h2>Ajouter une image</h2></div>
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
                <strong>Déposez une photo de la route ici</strong>
                <span>ou <u>choisissez un fichier</u> · JPG, PNG ou WebP · 20 Mo maximum</span>
              </div>
            )}
          </label>
          <div className="field-note"><ShieldAlert size={14} /> Utilisez une photo nette, bien éclairée, montrant des détails reconnaissables.</div>
        </div>
        <div className="panel run-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">LIEU</span><h2>Choisir la scène</h2></div>
            <span className="step-count">02 / 02</span>
          </div>
          <label className="field-label" htmlFor="scene-select">Scène correspondante</label>
          <div className="select-wrap">
            <select id="scene-select" value={sceneId} onChange={(event) => setSceneId(event.target.value)} disabled={!readyScenes.length || busy}>
              {!readyScenes.length && <option value="">Aucune scène disponible</option>}
              {readyScenes.map((scene) => <option value={scene.id} key={scene.id}>{scene.name}</option>)}
            </select>
            <ChevronDown size={16} />
          </div>
          <div className="method-note">
            <Sparkles size={15} />
            <p>Le résultat indique si la photo correspond suffisamment à la scène. Une position « À vérifier » reste approximative.</p>
          </div>
          {error && <div className="form-error">{error}</div>}
          <button className="button button-primary run-button" disabled={!file || !sceneId || busy}>
            {busy ? <><LoaderCircle className="spin" size={17} /> Analyse de la photo…</> : <><ScanSearch size={17} /> Trouver la position</>}
          </button>
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
          <div className="result-title">{item.status === "localized" ? "Position trouvée" : item.status === "uncertain" ? "Position à vérifier" : "Position introuvable"}</div>
          <div className="result-subtitle">{item.error || (item.status === "localized" ? "Les détails de la photo correspondent suffisamment à la scène." : "Essayez une photo plus nette montrant davantage le lieu.")}</div>
        </div>
        <StatusBadge status={item.status} />
      </div>
      {pose ? (
        <>
          {typeof item.metrics.calibration_warning === "string" && item.metrics.calibration_warning && (
            <div className="notice notice-warning calibration-notice"><ShieldAlert size={14} /><span>Le format de la photo diffère de celui prévu. Si elle a été recadrée, la position peut être moins fiable.</span></div>
          )}
          <div className="result-metrics">
            <div><span>DÉTAILS VISUELS RECONNUS</span><strong>{item.metrics.inliers ?? "—"}</strong><small>{typeof item.metrics.inlier_ratio === "number" ? `${(item.metrics.inlier_ratio * 100).toFixed(1)}% cohérents entre eux` : "Nombre de détails concordants"}</small></div>
          </div>
          <div className="pose-grid">
            <div className="pose-block"><span><Navigation size={13} /> POSITION ESTIMÉE · X, Y, Z</span>
              <strong>{pose.camera_center.map((value) => formatNumber(value, 3)).join(" , ")}</strong>
              <small>Unités relatives à la scène, pas des mètres vérifiés</small>
            </div>
            <div className="pose-block"><span><Radio size={13} /> ORIENTATION ESTIMÉE · X, Y, Z</span>
              <strong>{pose.euler_xyz_degrees.map((value) => `${formatNumber(value, 1)}°`).join(" , ")}</strong>
              <small>Angles en degrés</small>
            </div>
          </div>
          <details className="result-advanced">
            <summary>Détails supplémentaires</summary>
            <div>
              <span>Erreur d’alignement</span><strong>{formatNumber(item.metrics.median_reprojection_error_px)} px</strong>
              <span>Photo de référence la plus proche</span><strong>{item.metrics.reference_name || "—"}</strong>
              <span>Nombre de correspondances examinées</span><strong>{item.metrics.matches ?? "—"}</strong>
            </div>
          </details>
          <div className="result-footer">
            <span>{scene?.coordinate_frame_verified ? "Position affichable dans la scène 3D" : "Affichage 3D non vérifié"}</span>
            <button className="button button-secondary" onClick={() => onExplore(item)}>Voir dans la scène 3D <ArrowRight size={14} /></button>
          </div>
        </>
      ) : (
        <div className="failure-detail">
          <strong>Aucune position fiable n’a été trouvée.</strong>
          <span>Essayez une photo nette, bien éclairée, qui montre davantage de détails du lieu.</span>
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
      <PageTitle eyebrow="VOS RÉSULTATS" title="Historique" subtitle="Retrouvez les positions estimées pour vos photos." />
      <div className="panel history-panel">
        <div className="table-heading"><div><h2>Photos analysées</h2><span>{localizations.length} résultat(s)</span></div></div>
        {localizations.length ? (
          <div className="table-scroll">
            <table>
              <thead><tr><th>PHOTO</th><th>SCÈNE</th><th>RÉSULTAT</th><th>DÉTAILS RECONNUS</th><th>DATE</th><th /></tr></thead>
              <tbody>
                {localizations.map((item) => (
                  <tr key={item.id}>
                    <td><div className="image-cell"><div className="mini-image"><FileImage size={14} /></div><div><strong>{item.image_name}</strong><small className="mono">{item.id.slice(0, 8)}</small></div></div></td>
                    <td>{scenes.find((scene) => scene.id === item.scene_id)?.name || item.scene_id}</td>
                    <td><StatusBadge status={item.status} /></td>
                    <td className="mono">{item.metrics.inliers ?? "—"}</td>
                    <td className="table-date">{item.created_at ? new Date(item.created_at).toLocaleString("fr-FR") : "—"}</td>
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
      <PageTitle eyebrow="VUE 3D" title="Scène 3D" subtitle="Explorez la scène et consultez les positions trouvées." />
      {!scenes.length ? (
        <div className="panel"><EmptyState icon={Box} title="Aucune scène disponible" detail="Une scène doit être configurée avant de pouvoir l’afficher." /></div>
      ) : (
        <>
          <div className="explorer-toolbar panel">
            <label><span>SCÈNE</span><select value={sceneId} onChange={(event) => setSceneId(event.target.value)}>{scenes.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
            <label><span>POSITION</span><select value={active?.id || ""} onChange={(event) => setSelectedId(event.target.value)}><option value="">Aucune sélection</option>{sceneResults.map((item) => <option value={item.id} key={item.id}>{item.image_name} · {item.status === "localized" ? "trouvée" : "à vérifier"}</option>)}</select></label>
            <div className="toolbar-info"><span className="health-led led-online" /> {visiblePoses.length} position(s)</div>
          </div>
          {scene?.has_gaussian_ply && !scene.coordinate_frame_verified && (
            <div className="notice notice-warning"><ShieldAlert size={16} /><span>Les positions ne peuvent pas encore être affichées sur cette scène 3D : leur alignement n’a pas été vérifié.</span></div>
          )}
          <div className="explorer-layout">
            <div className="panel viewer-panel">
              <div className="viewer-heading"><div><span className="eyebrow">SCÈNE</span><h2>{scene?.name}</h2></div><div className="viewer-tools"><span><span className="legend-dot cloud" /> Décor 3D</span><span><span className="legend-dot camera" /> Position estimée</span></div></div>
              <Suspense fallback={<div className="viewer-shell viewer-loading">Loading 3D viewer…</div>}>
                <SceneViewer modelUrl={modelUrl} sceneId={scene?.id ?? null} poses={visiblePoses} overlayAllowed={Boolean(scene?.coordinate_frame_verified)} />
              </Suspense>
              {!scene?.has_gaussian_ply && <div className="viewer-footnote">Le modèle 3D n’est pas disponible pour cette scène.</div>}
            </div>
            <aside className="panel pose-sidebar">
              <div className="panel-heading"><div><span className="eyebrow">RÉSULTAT SÉLECTIONNÉ</span><h2>Position de la caméra</h2></div></div>
              {active?.pose ? (
                <div className="pose-details">
                  <StatusBadge status={active.status} />
                  <div className="pose-detail-image"><ImagePlus size={16} /><span>{active.image_name}</span></div>
                  <div className="detail-group"><span>POSITION · UNITÉS RELATIVES</span><strong>{active.pose.camera_center.map((value) => formatNumber(value, 4)).join("\n")}</strong></div>
                  <div className="detail-group"><span>ORIENTATION</span><strong>{active.pose.euler_xyz_degrees.map((value) => `${formatNumber(value, 2)}°`).join(" · ")}</strong></div>
                  <div className="detail-group"><span>DÉTAILS VISUELS RECONNUS</span><strong>{active.metrics.inliers ?? "—"}</strong></div>
                </div>
              ) : (
                <EmptyState icon={MapPin} title="Aucune position sélectionnée" detail="Choisissez une photo déjà analysée ou recherchez une nouvelle position." />
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
      content: "Je peux expliquer un résultat, comparer des positions enregistrées et vous aider à comprendre pourquoi une photo n’a pas donné de position fiable. Je m’appuie uniquement sur les données disponibles dans cet espace.",
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
      <PageTitle eyebrow="AIDE" title="Assistant" subtitle="Posez une question sur vos photos et les positions estimées." action={<div className="ai-ready"><span className="health-led" /> Assistant disponible</div>} />
      <div className="copilot-layout">
        <section className="panel chat-panel">
          <div className="chat-heading"><div className="copilot-avatar"><Bot size={18} /></div><div><strong>Assistant Fieldnotes</strong><span>Réponses fondées sur les résultats enregistrés</span></div><span className="online-label"><span className="health-led led-online" /> PRÊT</span></div>
          <div className="chat-messages">
            {messages.map((message, index) => (
              <div className={`chat-message message-${message.role}`} key={`${index}-${message.role}`}>
                <div className="message-avatar">{message.role === "assistant" ? <Sparkles size={14} /> : "Y"}</div>
                <div className="message-body"><span>{message.role === "assistant" ? "COPILOT" : "YOU"}</span><p>{message.content}</p></div>
              </div>
            ))}
            {sending && <div className="typing-state"><LoaderCircle className="spin" size={15} /> Recherche des résultats…</div>}
          </div>
          <form className="chat-composer" onSubmit={send}>
            {file && <div className="attachment-chip"><FileImage size={13} /> {file.name}<button type="button" aria-label="Remove attachment" onClick={() => setFile(null)}><X size={13} /></button></div>}
            <div className="composer-row">
              <label className="attach-button" title="Attach image"><input type="file" accept="image/*" onChange={(event) => setFile(event.target.files?.[0] || null)} /><ImagePlus size={17} /></label>
              <input value={text} onChange={(event) => setText(event.target.value)} placeholder="Question sur une photo ou une position…" />
              <button className="send-button" disabled={sending || (!text.trim() && !file)} aria-label="Send message"><Send size={16} /></button>
            </div>
            {file && (
              <div className="composer-scene">
                <span>Scène</span>
                <select value={sceneId} onChange={(event) => setSceneId(event.target.value)}>
                  <option value="">Choose a scene</option>
                  {scenes.filter((scene) => scene.ready).map((scene) => <option key={scene.id} value={scene.id}>{scene.name}</option>)}
                </select>
                <small>Demandez à l’assistant de trouver la position de cette photo.</small>
              </div>
            )}
            {error && <div className="form-error">{error}</div>}
            <div className="composer-footnote"><ShieldAlert size={12} /> Les réponses automatiques peuvent nécessiter une configuration par l’administrateur.</div>
          </form>
        </section>
        <aside className="copilot-aside">
          <div className="panel copilot-info">
            <div className="copilot-aside-icon"><MessageSquareText size={17} /></div>
            <h3>Je peux vous aider à</h3>
            <ul><li>Comprendre une position ou un résultat incertain.</li><li>Comparer des photos déjà analysées.</li><li>Vérifier si une scène est disponible.</li><li>Estimer la position d’une photo jointe sur demande.</li></ul>
          </div>
          <div className="notice notice-subtle"><ShieldAlert size={15} /><span>Les questions sont envoyées au service d’assistance configuré. Évitez les images confidentielles.</span></div>
        </aside>
      </div>
    </>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("dashboard");
  const [scenes, setScenes] = useState<Scene[]>([]);
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
    const [healthResult2, sceneResult2, localizationResult] = await Promise.allSettled([
      api.health(),
      api.scenes(),
      api.localizations(),
    ]);
    setHealth(healthResult2.status === "fulfilled" ? healthResult2.value.status : "unavailable");
    if (sceneResult2.status === "fulfilled") setScenes(sceneResult2.value);
    if (localizationResult.status === "fulfilled") setLocalizations(localizationResult.value);
    const failed = [sceneResult2, localizationResult].find((item) => item.status === "rejected");
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
        <div className="brand"><div className="brand-mark"><Navigation size={18} /></div><div><strong>FIELDNOTES</strong><span>POSITION VISUELLE</span></div></div>
        <div className="workspace-switch"><div className="workspace-dot">R</div><div><span>ESPACE</span><strong>Scène routière</strong></div><ChevronDown size={14} /></div>
        <div className="nav-section-label">MENU</div>
        <nav className="main-nav">
          {navItems.map(({ id, label, icon: Icon }) => (
            <button className={`nav-item ${page === id ? "nav-active" : ""}`} key={id} onClick={() => nav(id)}>
              <Icon size={17} strokeWidth={1.8} /><span>{label}</span>
              {id === "copilot" && <span className="nav-new"><Sparkles size={11} /></span>}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="backend-card"><div className="backend-card-title"><span className={`health-led ${health === "ok" ? "led-online" : ""}`} /> État du service</div><strong>{health === "ok" ? "Disponible" : health === "checking" ? "Connexion…" : "Indisponible"}</strong></div>
          <div className="sidebar-footer"><span>POSITION VISUELLE</span><span>Estimation expérimentale</span></div>
        </div>
      </aside>
      {mobileNavOpen && <button className="mobile-backdrop" aria-label="Close navigation" onClick={() => setMobileNavOpen(false)} />}
      <main className="main-shell">
        <header className="topbar">
          <button className="mobile-menu" aria-label="Open navigation" onClick={() => setMobileNavOpen(true)}><Menu size={19} /></button>
          <div className="breadcrumbs"><span>Espace</span><span className="breadcrumb-sep">/</span><strong>{currentLabel}</strong></div>
          <div className="topbar-right"><span className="live-tag"><span className="health-led led-online" /> ENVIRONNEMENT LOCAL</span><span className="topbar-divider" /><div className="avatar">RM</div></div>
        </header>
        <div className="page-content">
          {loadError && <div className="notice notice-error"><ShieldAlert size={16} /><span>Impossible de charger tous les résultats : {loadError}</span><button className="text-button" onClick={() => void refresh()}>Réessayer</button></div>}
          {latestResult && page === "localize" && (
            <div className="localization-result-wrap">
              <div className="result-wrap-heading"><span className="eyebrow">LATEST ATTEMPT</span><button className="icon-button" aria-label="Dismiss latest result" onClick={() => setLatestResult(null)}><X size={15} /></button></div>
              <ResultPanel item={latestResult} scene={scenes.find((scene) => scene.id === latestResult.scene_id)} onExplore={explore} />
            </div>
          )}
          {loading && page === "dashboard" ? <div className="loading-view"><LoaderCircle className="spin" size={19} /> Connecting to localization services…</div> : null}
          {page === "dashboard" && !loading && <Dashboard scenes={scenes} localizations={localizations} health={health} navigate={nav} />}
          {page === "localize" && (
            <LocalizationPage
              scenes={scenes}
              onComplete={handleComplete}
            />
          )}
          {page === "history" && <HistoryPage scenes={scenes} localizations={localizations} onExplore={explore} />}
          {page === "explorer" && <ExplorerPage scenes={scenes} localizations={localizations} selected={selectedLocalization} />}
          {page === "copilot" && <CopilotPage scenes={scenes} />}
          <footer className="page-footer"><span>Fieldnotes <span className="footer-dot">·</span> Position visuelle</span><span>Vérifiez les résultats marqués « À vérifier » avant de les utiliser.</span></footer>
        </div>
      </main>
      {toastVisible && latestResult && page === "localize" && (
        <div className="toast"><Check size={15} /> Résultat enregistré <button aria-label="Fermer la notification" onClick={() => setToastVisible(false)}><X size={14} /></button></div>
      )}
    </div>
  );
}
