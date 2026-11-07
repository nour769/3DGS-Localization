import type { DataStatus, Localization, LocalizationResult, Scene } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  const payload = (await response.json().catch(() => ({}))) as T & {
    error?: string;
  };
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return payload;
}

export const api = {
  health: () => request<{ status: string; database: string }>("/health"),
  scenes: async () => (await request<{ scenes: Scene[] }>("/scenes")).scenes,
  dataStatus: () => request<DataStatus>("/data-status"),
  localizations: async (limit = 100) =>
    (
      await request<{ localizations: Localization[] }>(
        `/localizations?limit=${limit}`,
      )
    ).localizations,
  localization: async (id: string) =>
    (
      await request<{ localization: Localization }>(`/localizations/${id}`)
    ).localization,
  visualize: (id: string) =>
    request<{
      localization_id: string;
      scene_id: string;
      pose: Localization["pose"];
      scene: Scene | null;
      overlay_allowed: boolean;
      coordinate_frame_warning: string | null;
    }>(`/localizations/${id}/visualization`),
  localize: async (file: File, sceneId: string) => {
    const form = new FormData();
    form.append("image", file);
    form.append("scene_id", sceneId);
    const response = await fetch(`${API_BASE}/localize`, {
      method: "POST",
      body: form,
    });
    const payload = (await response.json()) as LocalizationResult & {
      error?: string;
    };
    if (!response.ok && !payload.localization) {
      throw new Error(payload.error || `Request failed (${response.status})`);
    }
    return payload;
  },
  localizeConfiguredQuery: async (sceneId: string) => {
    const response = await fetch(`${API_BASE}/localize/configured-query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scene_id: sceneId }),
    });
    const payload = (await response.json()) as LocalizationResult & {
      error?: string;
    };
    if (!response.ok && !payload.localization) {
      throw new Error(payload.error || `Request failed (${response.status})`);
    }
    return payload;
  },
  copilot: async (
    messages: { role: "user" | "assistant"; content: string }[],
    image: File | null,
    sceneId: string,
  ) => {
    if (image) {
      const form = new FormData();
      form.append("messages", JSON.stringify(messages));
      form.append("scene_id", sceneId);
      form.append("image", image);
      return request<{ reply: string; provider_configured: boolean }>(
        "/copilot/chat",
        { method: "POST", body: form },
      );
    }
    return request<{ reply: string; provider_configured: boolean }>(
      "/copilot/chat",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages, scene_id: sceneId }),
      },
    );
  },
};

export function absoluteModelUrl(modelUrl: string) {
  if (/^https?:\/\//.test(modelUrl)) return modelUrl;
  if (API_BASE.startsWith("http")) {
    return new URL(modelUrl, API_BASE).toString();
  }
  return modelUrl;
}
