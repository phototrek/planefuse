// Minimal typed client over the focusstack-server REST API (SPEC §9).
export interface SystemInfo { device: string; device_name: string; free_memory: number; version: string; torch: string; }
export interface ParamSpec { name: string; label: string; type: string; default: unknown; min: number | null; max: number | null; choices: string[] | null; tooltip: string | null; }
export interface Algorithm { name: string; params: ParamSpec[]; }
export interface FileStatus { name: string; path: string; status: string; message: string; }
export interface ScanReport { ok: boolean; width: number | null; height: number | null; bit_depth: number | null; files: FileStatus[]; }
export interface Project { id: string; name: string; directory: string; frames: string[]; images: Record<string, Record<string, unknown>>; jobs: unknown[]; ui_state: Record<string, unknown>; }
export interface Job { id: string; type: string; status: string; percent: number; message: string; result: unknown; error: string; params: Record<string, unknown>; }
export interface FsEntry { name: string; path: string; is_dir: boolean; }
export interface FsList { path: string; entries: FsEntry[]; image_count: number; }
export interface RetouchSource { image_id: string; method: string | null; }
export interface RetouchSession {
  id: string;
  target_image_id: string;
  working_image_id: string;
  sources: string[];
  strokes: RetouchStroke[];
  rev: number;
}
export interface RetouchStroke {
  source_id: string;
  points: [number, number, number][];
  radius: number;
  hardness: number;
  opacity: number;
  mode: 'normal' | 'erase';
}
export interface RetouchMutation {
  rev: number;
  dirty_tiles: { z: number; x: number; y: number }[];
}

async function req<T>(method: string, url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: body !== undefined ? { 'content-type': 'application/json' } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined
  });
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const j = await r.json();
      detail = j.detail ?? j.error ?? detail;
    } catch {
      /* non-json body */
    }
    throw new Error(detail);
  }
  return r.status === 204 ? (undefined as T) : ((await r.json()) as T);
}

export const api = {
  system: () => req<SystemInfo>('GET', '/api/system'),
  algorithms: () => req<Algorithm[]>('GET', '/api/algorithms'),
  fsList: (path: string) => req<FsList>('GET', `/api/fs/list?path=${encodeURIComponent(path)}`),
  listProjects: () => req<Project[]>('GET', '/api/projects'),
  createProject: (path: string, name: string) => req<Project>('POST', '/api/projects', { path, name }),
  getProject: (id: string) => req<Project>('GET', `/api/projects/${id}`),
  scan: (id: string, path: string) => req<ScanReport>('POST', `/api/projects/${id}/frames/scan`, { path }),
  autoGroup: (id: string) => req<{ groups: string[][] }>('POST', `/api/projects/${id}/frames/auto-group`),
  enqueueStack: (id: string, params: Record<string, unknown>) =>
    req<{ id: string }>('POST', `/api/projects/${id}/jobs`, { type: 'stack', params }),
  enqueueJob: (id: string, type: string, params: Record<string, unknown>) =>
    req<{ id: string }>('POST', `/api/projects/${id}/jobs`, { type, params }),
  listJobs: () => req<Job[]>('GET', '/api/jobs'),
  getJob: (jid: string) => req<Job>('GET', `/api/jobs/${jid}`),
  cancelJob: (jid: string) => req('DELETE', `/api/jobs/${jid}`),
  registerView: (id: string, path: string) =>
    req<{ image_id: string; levels: number; width: number; height: number }>(
      'POST',
      `/api/projects/${id}/viewer/register`,
      { path }
    ),
  export: (id: string, body: Record<string, unknown>) =>
    req<{ id: string }>('POST', `/api/projects/${id}/export`, body),
  listPresets: () => req<{ name: string; params: Record<string, unknown> }[]>('GET', '/api/presets'),
  addPreset: (name: string, params: Record<string, unknown>) => req('POST', '/api/presets', { name, params }),
  deletePreset: (name: string) => req('DELETE', `/api/presets/${encodeURIComponent(name)}`),
  patchUiState: (id: string, state: Record<string, unknown>) =>
    req('PATCH', `/api/projects/${id}/ui-state`, state),
  listRetouch: (id: string) =>
    req<{ sessions: RetouchSession[] }>('GET', `/api/projects/${id}/retouch`),
  createRetouch: (id: string, targetImageId: string) =>
    req<{ session_id: string; working_image_id: string; sources: RetouchSource[] }>(
      'POST',
      `/api/projects/${id}/retouch`,
      { target_image_id: targetImageId }
    ),
  retouchStroke: (id: string, stroke: RetouchStroke) =>
    req<RetouchMutation>('POST', `/api/retouch/${id}/stroke`, stroke),
  retouchUndo: (id: string) =>
    req<RetouchMutation>('POST', `/api/retouch/${id}/undo`),
  retouchRedo: (id: string) =>
    req<RetouchMutation>('POST', `/api/retouch/${id}/redo`),
  flattenRetouch: (id: string, name: string) =>
    req<{ image_id: string }>('POST', `/api/retouch/${id}/flatten`, { name }),
  thumbUrl: (id: string, path: string) => `/api/projects/${id}/frame-thumb?path=${encodeURIComponent(path)}`,
  tileUrl: (imageId: string, z: number, x: number, y: number, rev?: number) =>
    `/api/viewer/${imageId}/tile/${z}/${x}/${y}${rev === undefined ? '' : `?rev=${rev}`}`
};
