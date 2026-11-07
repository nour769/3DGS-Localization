export type Status = "localized" | "uncertain" | "failed" | "processing";

export interface Scene {
  id: string;
  name: string;
  ready: boolean;
  has_gaussian_ply: boolean;
  coordinate_frame_verified: boolean;
  warning: string | null;
  model_url: string | null;
  reference_image_count?: number;
  colmap_files?: Record<string, boolean>;
  query_image_available?: boolean;
}

export interface SceneDataStatus {
  id: string;
  name: string;
  ready: boolean;
  ply_path: string | null;
  ply_available: boolean;
  colmap_model_path: string | null;
  colmap_files: Record<string, boolean>;
  reference_images_path: string | null;
  reference_image_count: number;
  query_image_path: string | null;
  query_image_available: boolean;
  query_calib_path: string | null;
  query_calib_available: boolean;
  missing: string | null;
  coordinate_frame_verified: boolean;
}

export interface DataStatus {
  project_data_root: string | null;
  scenes: SceneDataStatus[];
  notes: string[];
}

export interface Pose {
  coordinate_frame: string;
  translation_units: string;
  pose_convention: string;
  world_to_camera: number[][];
  rotation_matrix: number[][];
  translation: number[];
  camera_center: number[];
  quaternion_wxyz: number[];
  euler_xyz_degrees: number[];
}

export interface Metrics {
  inliers?: number;
  inlier_ratio?: number;
  matches?: number;
  mean_reprojection_error_px?: number;
  median_reprojection_error_px?: number;
  reference_name?: string;
  retrieval_method?: string;
  query_size?: number[];
  working_size?: number[];
  candidate_references?: string[];
  coordinate_scale_verified_metric?: boolean;
  gaussian_colmap_alignment_verified?: boolean;
  [key: string]: unknown;
}

export interface Localization {
  id: string;
  image_name: string;
  scene_id: string;
  status: Status;
  pose: Pose | null;
  metrics: Metrics;
  error: string | null;
  created_at: string | null;
}

export interface LocalizationResult {
  localization: Localization;
  result?: {
    status: Status;
    message?: string | null;
    pose: Pose | null;
    metrics: Metrics;
  };
  error?: string;
}
