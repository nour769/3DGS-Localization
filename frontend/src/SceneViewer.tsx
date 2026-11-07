import { useEffect, useRef, useState } from "react";
import {
  Color,
  Entity,
  Quat,
  StandardMaterial,
  Vec3,
} from "playcanvas";
import { createViewer } from "@playcanvas/supersplat-viewer/viewer";
import "@playcanvas/supersplat-viewer/viewer.css";
import { defaultSettings } from "@playcanvas/supersplat-viewer/settings";
import type { Pose } from "./types";

interface SceneViewerProps {
  modelUrl: string | null;
  sceneId: string | null;
  poses: Pose[];
  overlayAllowed: boolean;
}

interface LoadedSplatInfo {
  splatCount: number;
  shDegree: number;
}

type SuperSplatViewer = Awaited<ReturnType<typeof createViewer>>;

interface ViewerRuntime {
  viewer: SuperSplatViewer;
  poseRoot: Entity;
  radius: number;
  materials: StandardMaterial[];
}

function worldPointFromCamera(
  pose: Pose,
  x: number,
  y: number,
  z: number,
): Vec3 {
  const center = pose.camera_center as [number, number, number];
  const rotation = pose.rotation_matrix;
  const cameraY = -y;
  const cameraZ = -z;
  return new Vec3(
    center[0] +
      rotation[0][0] * x +
      rotation[1][0] * cameraY +
      rotation[2][0] * cameraZ,
    center[1] +
      rotation[0][1] * x +
      rotation[1][1] * cameraY +
      rotation[2][1] * cameraZ,
    center[2] +
      rotation[0][2] * x +
      rotation[1][2] * cameraY +
      rotation[2][2] * cameraZ,
  );
}

function addCameraMarker(
  parent: Entity,
  pose: Pose,
  color: Color,
  radius: number,
): StandardMaterial {
  const center = new Vec3(...(pose.camera_center as [number, number, number]));
  const halfWidth = radius * 0.035;
  const halfHeight = radius * 0.021;
  const depth = radius * 0.07;
  const corners = [
    worldPointFromCamera(pose, -halfWidth, -halfHeight, -depth),
    worldPointFromCamera(pose, halfWidth, -halfHeight, -depth),
    worldPointFromCamera(pose, halfWidth, halfHeight, -depth),
    worldPointFromCamera(pose, -halfWidth, halfHeight, -depth),
  ];
  const material = new StandardMaterial();
  material.diffuse = color.clone();
  material.emissive = color.clone();
  material.update();

  const marker = new Entity("localized-camera");
  marker.addComponent("render", { type: "sphere", material });
  marker.setPosition(center);
  marker.setLocalScale(radius * 0.012, radius * 0.012, radius * 0.012);
  parent.addChild(marker);

  const addSegment = (start: Vec3, end: Vec3) => {
    const direction = end.clone().sub(start);
    const length = direction.length();
    if (!length) return;
    const segment = new Entity("camera-frustum");
    segment.addComponent("render", { type: "box", material });
    segment.setPosition(start.clone().add(end).mulScalar(0.5));
    segment.setLocalScale(radius * 0.0015, length, radius * 0.0015);
    segment.setRotation(
      new Quat().setFromDirections(Vec3.UP, direction.normalize()),
    );
    parent.addChild(segment);
  };

  corners.forEach((corner, index) => {
    addSegment(center, corner);
    addSegment(corner, corners[(index + 1) % corners.length]);
  });
  return material;
}

export default function SceneViewer({
  modelUrl,
  sceneId,
  poses,
  overlayAllowed,
}: SceneViewerProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const runtimeRef = useRef<ViewerRuntime | null>(null);
  const poseStateRef = useRef({ poses, overlayAllowed });
  const refreshOverlaysRef = useRef<() => void>(() => undefined);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [cameraWarning, setCameraWarning] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [splatInfo, setSplatInfo] = useState<LoadedSplatInfo | null>(null);
  poseStateRef.current = { poses, overlayAllowed };

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    setLoadError(null);
    setCameraWarning(null);
    setLoaded(false);
    setSplatInfo(null);
    runtimeRef.current = null;

    if (!modelUrl) {
      return;
    }

    let active = true;
    let viewer: SuperSplatViewer | null = null;

    const onLoaded = () => {
      if (!active || !viewer || !viewer.state.loaded) return;
      const splatComponent = viewer.app.root.findComponents("gsplat")[0];
      const resource = splatComponent?.resource as {
        gsplatData?: { numSplats: number; shBands: number };
        aabb?: { halfExtents: Vec3 };
      } | null;
      if (!resource?.gsplatData || !resource.aabb) {
        setLoadError("SuperSplat loaded no Gaussian data or scene bounds.");
        return;
      }

      const poseRoot = new Entity("localization-camera-poses");
      viewer.app.root.addChild(poseRoot);
      runtimeRef.current = {
        viewer,
        poseRoot,
        radius: Math.max(resource.aabb.halfExtents.length(), 0.1),
        materials: [],
      };
      setSplatInfo({
        splatCount: resource.gsplatData.numSplats,
        shDegree: resource.gsplatData.shBands,
      });
      setLoaded(true);
      refreshOverlaysRef.current();
    };

    void (async () => {
      try {
        const settings = defaultSettings("environment");
        if (sceneId) {
          try {
            const response = await fetch(
              `/api/scenes/${encodeURIComponent(sceneId)}/viewer-camera`,
            );
            if (!response.ok) {
              const details = (await response.json()) as { error?: string };
              throw new Error(
                details.error || `Camera framing request failed (${response.status}).`,
              );
            }
            const view = (await response.json()) as {
              position: [number, number, number];
              target: [number, number, number];
              fov: number;
            };
            settings.cameras = [
              {
                initial: {
                  position: view.position,
                  target: view.target,
                  fov: view.fov,
                },
              },
            ];
          } catch (error) {
            setCameraWarning(
              error instanceof Error
                ? `COLMAP camera framing unavailable: ${error.message}`
                : "COLMAP camera framing unavailable; using the viewer default.",
            );
          }
        }
        viewer = await createViewer({
          container: host,
          contentUrl: modelUrl,
          contentFilename: "scene.ply",
          settings,
          ui: false,
        });
        if (!active) {
          viewer.destroy();
          return;
        }
        viewer.events.on("loaded:changed", onLoaded);
        onLoaded();
      } catch (error) {
        if (active) {
          setLoadError(
            error instanceof Error ? error.message : "Could not load the Gaussian PLY.",
          );
        }
      }
    })();

    return () => {
      active = false;
      refreshOverlaysRef.current = () => undefined;
      runtimeRef.current = null;
      viewer?.destroy();
    };
  }, [modelUrl, sceneId]);

  useEffect(() => {
    refreshOverlaysRef.current = () => {
      const runtime = runtimeRef.current;
      if (!runtime) return;
      while (runtime.poseRoot.children.length) {
        runtime.poseRoot.children[0].destroy();
      }
      runtime.materials.forEach((material) => material.destroy());
      runtime.materials = [];
      const poseState = poseStateRef.current;
      if (!poseState.overlayAllowed) return;
      poseState.poses.forEach((pose, index) => {
        runtime.materials.push(
          addCameraMarker(
            runtime.poseRoot,
            pose,
            new Color(
              index === 0 ? 0.25 : 0.9,
              index === 0 ? 0.82 : 0.58,
              0.9,
            ),
            runtime.radius,
          ),
        );
      });
    };
    refreshOverlaysRef.current();
  }, [poses, overlayAllowed, loaded, splatInfo]);

  return (
    <div className="viewer-shell">
      <div className="viewer-canvas" ref={hostRef} />
      {modelUrl && !loaded && !loadError && (
        <div className="viewer-overlay">Loading SuperSplat Gaussian scene…</div>
      )}
      {loaded && splatInfo && (
        <div className="viewer-overlay">
          SuperSplat 3DGS · {splatInfo.splatCount.toLocaleString()} splats · SH
          {" "}degree {splatInfo.shDegree}
        </div>
      )}
      {cameraWarning && (
        <div className="viewer-overlay viewer-camera-warning">{cameraWarning}</div>
      )}
      {loadError && (
        <div className="viewer-overlay viewer-error">
          Could not load the scene model: {loadError}
        </div>
      )}
      {!modelUrl && (
        <div className="viewer-overlay viewer-hint">
          Configure a Gaussian PLY to explore the scene
        </div>
      )}
      {modelUrl && (
        <div className="viewer-overlay viewer-hint">
          Drag to orbit · Shift/right-drag to pan · scroll to zoom
        </div>
      )}
    </div>
  );
}
