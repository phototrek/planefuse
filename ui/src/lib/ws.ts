// Single /ws connection feeding job events into appState.jobs, with reconnect.
import { appState } from './stores.svelte';
import type { Job } from './api';

let socket: WebSocket | null = null;
let backoff = 500;

interface JobEvent {
  job_id: string;
  type: string;
  status: string;
  percent: number;
  message: string;
}

export function connectJobs(): void {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }
  socket = new WebSocket(`ws://${location.host}/ws`);

  socket.onopen = () => {
    backoff = 500;
  };

  socket.onmessage = (ev) => {
    let e: JobEvent;
    try {
      e = JSON.parse(ev.data);
    } catch {
      return;
    }
    if (!e.job_id) return;
    const prev = appState.jobs[e.job_id];
    const merged: Job = {
      id: e.job_id,
      type: e.type ?? prev?.type ?? '',
      status: e.status ?? prev?.status ?? 'running',
      percent: e.percent ?? prev?.percent ?? 0,
      message: e.message ?? prev?.message ?? '',
      result: prev?.result ?? null,
      error: prev?.error ?? ''
    };
    appState.jobs[e.job_id] = merged;
  };

  socket.onclose = () => {
    socket = null;
    backoff = Math.min(backoff * 2, 8000);
    setTimeout(connectJobs, backoff);
  };

  socket.onerror = () => {
    socket?.close();
  };
}
