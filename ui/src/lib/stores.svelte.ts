import type { Job, Project, SystemInfo } from './api';

export const appState = $state<{
  system: SystemInfo | null;
  project: Project | null;
  jobs: Record<string, Job>;
}>({ system: null, project: null, jobs: {} });
