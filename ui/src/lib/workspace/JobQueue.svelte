<script lang="ts">
  import { tick } from 'svelte';
  import { api, type Job } from '$lib/api';
  import { appState } from '$lib/stores.svelte';
  import ProgressBar from '$lib/components/ProgressBar.svelte';

  type DisplayItem = {
    label: string;
    value: string;
    detail?: string;
  };

  type DisplayGroup = {
    title: string;
    items: DisplayItem[];
  };

  const labels: Record<string, string> = {
    algo_params: 'Stacking',
    align: 'Alignment',
    bit_depth: 'Bit depth',
    compression: 'Compression',
    contrast_threshold: 'Contrast threshold',
    correlation_threshold: 'Minimum quality',
    depth_dest: 'Depth map',
    dest: 'Destination',
    device: 'Processor',
    drop_misaligned: 'Exclude low-quality frames',
    estimation_radius: 'Estimation radius',
    float_tiff_dest: 'Scene-linear TIFF',
    format: 'File format',
    frames: 'Source photos',
    grid_cols: 'Grid columns',
    grid_rows: 'Grid rows',
    image_id: 'Source image',
    inner_method: 'Per-slab method',
    interp: 'Interpolation',
    jpeg_quality: 'JPEG quality',
    kurtosis_threshold: 'Kurtosis threshold',
    max_long_edge: 'Maximum long edge',
    method: 'Stacking method',
    model: 'Motion model',
    normalize_brightness: 'Match brightness',
    outer_method: 'Final method',
    select: 'Frame selection',
    selection_smoothing: 'Halo reduction',
    sharpness_radius: 'Sharpness radius',
    slab_overlap: 'Slab overlap',
    slab_size: 'Photos per slab',
    smoothing_radius: 'Smoothing radius',
    temperature: 'Blend softness'
  };

  const values: Record<string, string> = {
    auto: 'Automatic',
    bicubic: 'Bicubic',
    bilinear: 'Bilinear',
    cpu: 'CPU',
    cuda: 'NVIDIA GPU',
    dmap: 'Depth map',
    jpg: 'JPEG',
    lanczos3: 'Lanczos 3',
    mps: 'Apple GPU',
    perspective: 'Perspective',
    pmax: 'PMax',
    similarity: 'Similarity',
    slab: 'Large stack',
    tif: 'TIFF',
    translation: 'Translation',
    weighted: 'Weighted blend'
  };

  let jobs = $derived(Object.values(appState.jobs));
  let pending = $derived(jobs.filter((job) => job.status === 'pending'));
  let activeJob = $state<Job | null>(null);
  let dialog = $state<HTMLDialogElement>();
  let closeButton = $state<HTMLButtonElement>();
  let returnFocus: HTMLButtonElement | null = null;
  let displayGroups = $derived(activeJob ? groupsFor(activeJob.params) : []);

  function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
  }

  function humanizeKey(key: string): string {
    if (labels[key]) return labels[key];
    const words = key.replace(/[_-]+/g, ' ');
    return words.charAt(0).toUpperCase() + words.slice(1);
  }

  function fileName(path: string): string {
    return path.split(/[\\/]/).filter(Boolean).pop() ?? path;
  }

  function formatValue(key: string, value: unknown): DisplayItem {
    const label = humanizeKey(key);
    if (key === 'frames' && Array.isArray(value)) {
      const paths = value.filter((item): item is string => typeof item === 'string');
      const first = paths[0] ? fileName(paths[0]) : '';
      const last = paths.length > 1 ? fileName(paths[paths.length - 1]) : '';
      return {
        label,
        value: `${value.length} photo${value.length === 1 ? '' : 's'}`,
        detail: [first, last].filter(Boolean).join(' → ') || undefined
      };
    }
    if (Array.isArray(value)) {
      return {
        label,
        value: value.length
          ? value.map((item) => formatScalar(item)).join(', ')
          : 'None'
      };
    }
    return { label, value: formatScalar(value) };
  }

  function formatScalar(value: unknown): string {
    if (value === null || value === undefined || value === '') return 'Not set';
    if (typeof value === 'boolean') return value ? 'Yes' : 'No';
    if (typeof value === 'number') return new Intl.NumberFormat().format(value);
    const text = String(value);
    if (values[text.toLowerCase()]) return values[text.toLowerCase()];
    if (/^[a-z][a-z0-9_-]*$/i.test(text) && !text.includes('/') && !text.includes('\\')) {
      const words = text.replace(/[_-]+/g, ' ');
      return words.charAt(0).toUpperCase() + words.slice(1);
    }
    return text;
  }

  function nestedItems(params: Record<string, unknown>, prefix = ''): DisplayItem[] {
    const items: DisplayItem[] = [];
    for (const [key, value] of Object.entries(params)) {
      const label = [prefix, humanizeKey(key)].filter(Boolean).join(' · ');
      if (isRecord(value)) {
        const nested = nestedItems(value, label);
        items.push(...(nested.length ? nested : [{ label, value: 'Enabled' }]));
      } else {
        items.push({ ...formatValue(key, value), label });
      }
    }
    return items;
  }

  function groupsFor(params: Record<string, unknown>): DisplayGroup[] {
    const overview: DisplayItem[] = [];
    const groups: DisplayGroup[] = [];

    for (const [key, value] of Object.entries(params)) {
      if (isRecord(value)) {
        const items = nestedItems(value);
        groups.push({
          title: humanizeKey(key),
          items: items.length ? items : [{ label: 'State', value: 'Enabled' }]
        });
      } else {
        overview.push(formatValue(key, value));
      }
    }

    if (overview.length) groups.unshift({ title: 'Overview', items: overview });
    if (!groups.length) {
      groups.push({
        title: 'Overview',
        items: [{ label: 'Settings', value: 'No custom settings' }]
      });
    }
    return groups;
  }

  function jobTitle(type: string): string {
    return {
      export: 'Export settings',
      select: 'Frame selection settings',
      stack: 'Focus stack settings'
    }[type] ?? `${humanizeKey(type)} settings`;
  }

  async function showSettings(job: Job, trigger: HTMLButtonElement) {
    activeJob = job;
    returnFocus = trigger;
    await tick();
    dialog?.showModal();
    closeButton?.focus();
  }

  function closeSettings() {
    dialog?.close();
  }

  function finishClose() {
    activeJob = null;
    returnFocus?.focus();
    returnFocus = null;
  }

  function closeFromBackdrop(event: MouseEvent) {
    if (!dialog || event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    const outside = (
      event.clientX < bounds.left ||
      event.clientX > bounds.right ||
      event.clientY < bounds.top ||
      event.clientY > bounds.bottom
    );
    if (outside) closeSettings();
  }

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
        <button
          class="settings-button"
          data-testid="job-settings"
          data-job-id={job.id}
          onclick={(event) => showSettings(job, event.currentTarget)}
        >Settings</button>
      </div>
    </article>
  {/each}
</div>

{#if activeJob}
  <dialog
    bind:this={dialog}
    aria-labelledby="job-settings-title"
    aria-describedby="job-settings-description"
    data-testid="job-settings-dialog"
    data-job-id={activeJob.id}
    onclick={closeFromBackdrop}
    onclose={finishClose}
  >
    <div class="dialog-card">
      <header class="dialog-header">
        <div>
          <p class="dialog-kicker">Run settings</p>
          <h2 id="job-settings-title">{jobTitle(activeJob.type)}</h2>
          <p id="job-settings-description">
            Everything used for this job, grouped for quick review.
          </p>
        </div>
        <div class="dialog-header-actions">
          <span class="dialog-status" class:error={activeJob.status === 'error'}>
            {activeJob.status}
          </span>
          <button
            bind:this={closeButton}
            class="close-button"
            aria-label="Close settings"
            onclick={closeSettings}
          >×</button>
        </div>
      </header>

      <div class="parameter-groups">
        {#each displayGroups as group}
          <section class="parameter-group" class:wide={group.items.length > 4}>
            <h3>{group.title}</h3>
            <dl class="parameter-list">
              {#each group.items as item}
                <div class="parameter-item">
                  <dt>{item.label}</dt>
                  <dd>
                    <span>{item.value}</span>
                    {#if item.detail}<small>{item.detail}</small>{/if}
                  </dd>
                </div>
              {/each}
            </dl>
          </section>
        {/each}
      </div>
    </div>
  </dialog>
{/if}

<style>
  .queue { display: flex; gap: 8px; }
  .job { width: 180px; flex: 0 0 auto; display: flex; flex-direction: column; gap: 5px; padding: 8px; border: 1px solid var(--accent); border-radius: var(--radius); background: var(--panel-2); }
  .job.finished { border-color: var(--line); opacity: .78; }
  .heading { display: flex; justify-content: space-between; font-size: 10px; text-transform: uppercase; }
  .status { color: var(--text-faint); }
  .status.error { color: var(--bad); }
  .message { font-size: 9px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .actions { display: flex; gap: 4px; align-items: flex-start; }
  .actions button { padding: 3px 5px; font-size: 9px; }
  .settings-button { margin-left: auto; }

  dialog {
    width: min(780px, calc(100vw - 24px));
    max-width: none;
    max-height: calc(100dvh - 24px);
    margin: auto;
    padding: 0;
    overflow: hidden;
    border: 1px solid var(--line-strong);
    border-radius: 12px;
    background: var(--panel);
    color: var(--text);
    box-shadow: 0 28px 90px rgba(0, 0, 0, .72);
  }

  dialog::backdrop {
    background: rgba(4, 5, 7, .76);
    backdrop-filter: blur(4px);
  }

  .dialog-card {
    min-width: 0;
    background:
      radial-gradient(circle at 90% 0, rgba(255, 140, 66, .09), transparent 18rem),
      var(--panel);
  }

  .dialog-header {
    display: flex;
    min-width: 0;
    align-items: flex-start;
    justify-content: space-between;
    gap: 18px;
    padding: 16px 18px 14px;
    border-bottom: 1px solid var(--line);
  }

  .dialog-header > div:first-child { min-width: 0; }
  .dialog-kicker {
    margin: 0 0 5px;
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
  }
  .dialog-header h2 {
    margin: 0;
    font-size: 19px;
    font-weight: 650;
    letter-spacing: -.02em;
  }
  .dialog-header p:last-child {
    margin: 4px 0 0;
    color: var(--text-dim);
    font-size: 11px;
    line-height: 1.4;
  }

  .dialog-header-actions {
    display: flex;
    flex: 0 0 auto;
    align-items: center;
    gap: 8px;
  }
  .dialog-status {
    padding: 3px 7px;
    border: 1px solid color-mix(in srgb, var(--good) 35%, var(--line));
    border-radius: 999px;
    color: var(--good);
    font-family: var(--font-mono);
    font-size: 9px;
    text-transform: uppercase;
  }
  .dialog-status.error {
    border-color: color-mix(in srgb, var(--bad) 40%, var(--line));
    color: var(--bad);
  }
  .close-button {
    display: grid;
    width: 28px;
    height: 28px;
    padding: 0;
    place-items: center;
    border-color: var(--line);
    color: var(--text-dim);
    font-size: 18px;
    line-height: 1;
  }

  .parameter-groups {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(220px, 100%), 1fr));
    gap: 9px;
    min-width: 0;
    padding: 12px;
  }

  .parameter-group {
    min-width: 0;
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: 7px;
    background: color-mix(in srgb, var(--panel-2) 88%, transparent);
  }
  .parameter-group.wide { grid-column: 1 / -1; }
  .parameter-group h3 {
    margin: 0;
    padding: 7px 9px;
    border-bottom: 1px solid var(--line);
    color: var(--text-dim);
    font-family: var(--font-mono);
    font-size: 9px;
    font-weight: 650;
    letter-spacing: .09em;
    text-transform: uppercase;
  }

  .parameter-list {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(125px, 100%), 1fr));
    min-width: 0;
    margin: 0;
  }
  .parameter-item {
    min-width: 0;
    padding: 7px 9px 8px;
    border-right: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
  }
  .parameter-item dt {
    margin: 0 0 3px;
    color: var(--text-faint);
    font-size: 8px;
    font-weight: 650;
    letter-spacing: .04em;
    line-height: 1.25;
    text-transform: uppercase;
  }
  .parameter-item dd {
    min-width: 0;
    margin: 0;
    color: var(--text);
    font-size: 11px;
    font-weight: 550;
    line-height: 1.3;
    overflow-wrap: anywhere;
    word-break: break-word;
  }
  .parameter-item dd span,
  .parameter-item dd small {
    display: block;
    min-width: 0;
    overflow-wrap: anywhere;
  }
  .parameter-item dd small {
    margin-top: 2px;
    color: var(--text-faint);
    font-family: var(--font-mono);
    font-size: 8px;
    font-weight: 400;
    line-height: 1.3;
  }

  @media (max-width: 480px) {
    dialog { width: calc(100vw - 16px); }
    .dialog-header { padding: 12px; }
    .dialog-header p:last-child { display: none; }
    .parameter-groups {
      grid-template-columns: 1fr;
      gap: 6px;
      padding: 8px;
    }
    .parameter-list { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }

  @media (max-height: 560px) {
    .dialog-header { padding-block: 9px; }
    .dialog-header p:last-child { display: none; }
    .parameter-groups { gap: 6px; padding: 8px; }
    .parameter-group h3 { padding-block: 5px; }
    .parameter-item { padding-block: 5px; }
  }
</style>
