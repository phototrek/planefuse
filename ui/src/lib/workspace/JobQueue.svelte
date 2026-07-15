<script lang="ts">
  import { api, type Job } from '$lib/api';
  import { appState } from '$lib/stores.svelte';
  import ProgressBar from '$lib/components/ProgressBar.svelte';

  let jobs = $derived(Object.values(appState.jobs));
  let pending = $derived(jobs.filter((job) => job.status === 'pending'));

  async function cancel(job: Job) {
    await api.cancelJob(job.id);
  }

  async function move(job: Job, direction: -1 | 1) {
    const order = pending.map((item) => item.id);
    const index = order.indexOf(job.id);
    const target = index + direction;
    if (index < 0 || target < 0 || target >= order.length) return;
    [order[index], order[target]] = [order[target], order[index]];
    await api.reorderJobs(order);
  }

  async function rerun(job: Job) {
    if (!appState.project || job.type !== 'stack') return;
    const response = await api.enqueueJob(appState.project.id, job.type, job.params);
    appState.stackJobIds = [...appState.stackJobIds, response.id];
    appState.drawerOpen = true;
  }
</script>

<div class="queue" data-testid="job-queue">
  {#each jobs as job (job.id)}
    <article class="job" class:finished={!['pending', 'running'].includes(job.status)}>
      <div class="heading">
        <strong class="mono">{job.type}</strong>
        <span class="status" class:error={job.status === 'error'}>{job.status}</span>
      </div>
      <ProgressBar percent={job.percent} status={job.status} />
      <span class="message mono faint">{job.error || job.message || job.id}</span>
      <div class="actions">
        {#if job.status === 'pending'}
          <button aria-label="Move job earlier" onclick={() => move(job, -1)}>←</button>
          <button aria-label="Move job later" onclick={() => move(job, 1)}>→</button>
        {/if}
        {#if ['pending', 'running'].includes(job.status)}
          <button onclick={() => cancel(job)}>Cancel</button>
        {:else if job.type === 'stack'}
          <button onclick={() => rerun(job)}>Run again</button>
        {/if}
        <details>
          <summary>Parameters</summary>
          <pre>{JSON.stringify(job.params, null, 2)}</pre>
        </details>
      </div>
    </article>
  {/each}
</div>

<style>
  .queue { display: flex; gap: 8px; }
  .job { width: 180px; flex: 0 0 auto; display: flex; flex-direction: column; gap: 5px; padding: 8px; border: 1px solid var(--accent); border-radius: var(--radius); background: var(--panel-2); }
  .job.finished { border-color: var(--line); opacity: .78; }
  .heading { display: flex; justify-content: space-between; font-size: 10px; text-transform: uppercase; }
  .status { color: var(--text-faint); }
  .status.error { color: var(--bad); }
  .message { font-size: 9px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .actions { display: flex; gap: 4px; align-items: flex-start; }
  button, summary { font-size: 9px; padding: 3px 5px; }
  details { position: relative; }
  pre { position: absolute; bottom: 100%; right: 0; z-index: 20; width: 280px; max-height: 180px; overflow: auto; padding: 8px; background: var(--bg); border: 1px solid var(--line); font-size: 9px; }
</style>
