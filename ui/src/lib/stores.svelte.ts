import type { FileStatus, Job, Project, SystemInfo } from './api';

export interface WorkspaceResult {
  id: string;          // image_id key in project.images
  label: string;       // name or method
  method: string;
  path: string;
  imageId?: string;    // registered viewer image id (lazy)
  thumb?: string;      // tile-0 url (lazy)
  levels?: number;
  width?: number;
  height?: number;
}

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
  results: WorkspaceResult[];
  viewer: ViewerTarget;
  drawerOpen: boolean;
  stackJobIds: string[];      // enqueued stack jobs to watch for result discovery
  exportJobId: string | null; // current export job, for the export-done indicator
}>({
  system: null,
  project: null,
  jobs: {},
  inputs: [],
  results: [],
  viewer: null,
  drawerOpen: true,
  stackJobIds: [],
  exportJobId: null
});
