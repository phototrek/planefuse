<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { api, type Job } from '$lib/api';
  import { appState } from '$lib/stores.svelte';
  import { connectJobs } from '$lib/ws';
  import ProgressBar from '$lib/components/ProgressBar.svelte';

  const ACTIVE = new Set(['pending', 'running']);
  let poll: ReturnType<typeof setInterval> | null = null;

  async function seed() {
    try {
      for (const j of await api.listJobs()) appState.jobs[j.id] = j;
    } catch {
      /* server may be momentarily unavailable */
    }
  }

  // Fallback poll for active jobs in case the socket is down.
  async function pollActive() {
    const active = Object.values(appState.jobs).filter((j) => ACTIVE.has(j.status));
    for (const j of active) {
      try {
        appState.jobs[j.id] = await api.getJob(j.id);
      } catch {
        /* ignore */
      }
    }
  }

  onMount(() => {
    connectJobs();
    seed();
    poll = setInterval(pollActive, 1500);
  });
  onDestroy(() => poll && clearInterval(poll));

  let jobs = $derived(
    Object.values(appState.jobs).sort((a, b) => {
      const rank = (j: Job) => (ACTIVE.has(j.status) ? 0 : 1);
      return rank(a) - rank(b);
    })
  );

  async function cancel(id: string) {
    try {
      await api.cancelJob(id);
    } catch {
      /* already finished */
    }
  }

  const DONE = new Set(['done', 'error', 'cancelled']);

  async function rerun(job: Job) {
    if (!appState.project) return;
    try {
      await api.enqueueJob(appState.project.id, job.type, job.params ?? {});
    } catch {
      /* surfaced on the job itself */
    }
  }
</script>

<div class="screen">
  <header class="head">
    <p class="eyebrow">Screen 3</p>
    <h1>Queue &amp; Progress</h1>
  </header>

  {#if jobs.length === 0}
    <p class="dim">No jobs yet. Enqueue one from the <a href="/stack">Stack</a> screen.</p>
  {:else}
    <ul class="jobs">
      {#each jobs as j (j.id)}
        <li class="panel job">
          <div class="top">
            <span class="jtype mono">{j.type}</span>
            <span class="jid mono faint">{j.id}</span>
            <span class="jstatus" class:done={j.status === 'done'} class:error={j.status === 'error'} data-testid="job-status">
              {j.status}
            </span>
            {#if j.status === 'pending' || j.status === 'running'}
              <button class="ghost" onclick={() => cancel(j.id)}>Cancel</button>
            {:else if DONE.has(j.status) && appState.project}
              <button class="ghost" data-testid="rerun" onclick={() => rerun(j)} title="Re-run with the same parameters">↻ Re-run</button>
            {/if}
          </div>
          <ProgressBar percent={j.percent} status={j.status} />
          <div class="log mono faint">{j.message || (j.error ? `error: ${j.error}` : '')}</div>
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .screen { padding: 28px 32px; max-width: 900px; display: flex; flex-direction: column; gap: 18px; }
  .head h1 { font-size: 26px; margin-top: 4px; }
  .jobs { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 12px; }
  .job { padding: 14px 16px; display: flex; flex-direction: column; gap: 10px; }
  .top { display: flex; align-items: center; gap: 12px; }
  .jtype { text-transform: uppercase; font-size: 12px; letter-spacing: 0.06em; color: var(--accent-bright); }
  .jid { font-size: 11px; }
  .jstatus { margin-left: auto; font-size: 12px; font-weight: 600; color: var(--text-dim); text-transform: uppercase; }
  .jstatus.done { color: var(--good); }
  .jstatus.error { color: var(--bad); }
  .log { font-size: 11px; min-height: 14px; }
</style>
