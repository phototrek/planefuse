import type { FileStatus, Job, Project, ScanReport, SystemInfo } from './api';

export interface WorkspaceResult {
  id: string;          // image_id key in project.images
  label: string;       // name or method
  method: string;
  path: string;
  frames?: number;     // frame count from result metadata (for {frames} token)
  imageId?: string;    // registered viewer image id (lazy)
  thumb?: string;      // tile-0 url (lazy)
  levels?: number;
  width?: number;
  height?: number;
  domain: string;
  storage: string;
  metadata?: Record<string, unknown>;
  decoder: Record<string, unknown>;
  provenance: Record<string, unknown>;
}

export interface ViewerTransform { scale: number; tx: number; ty: number; }
export type CompareMode = 'single' | 'split-source' | 'side-by-side' | 'before-after';

export type ViewerTarget =
  | { kind: 'input'; path: string }
  | { kind: 'result'; id: string }
  | { kind: 'job'; id: string }
  | null;

export const appState = $state<{
  system: SystemInfo | null;
  project: Project | null;
  jobs: Record<string, Job>;
  inputs: FileStatus[];
  scanReport: ScanReport | null;
  results: WorkspaceResult[];
  viewer: ViewerTarget;
  drawerOpen: boolean;
  stackJobIds: string[];      // enqueued stack jobs to watch for result discovery
  exportJobId: string | null; // current export job, for the export-done indicator
  compareMode: CompareMode;
  compareResultId: string | null;
  viewerTransform: ViewerTransform | null;
  showHistogram: boolean;
  displayTonemap: boolean;
}>({
  system: null,
  project: null,
  jobs: {},
  inputs: [],
  scanReport: null,
  results: [],
  viewer: null,
  drawerOpen: true,
  stackJobIds: [],
  exportJobId: null,
  compareMode: 'single',
  compareResultId: null,
  viewerTransform: null,
  showHistogram: true,
  displayTonemap: true
});
