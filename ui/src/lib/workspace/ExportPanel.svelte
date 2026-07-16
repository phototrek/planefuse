<script lang="ts">
  import { api } from '$lib/api';
  import { appState } from '$lib/stores.svelte';
  import { applyTemplate, todayISO, type TemplateCtx } from './template';
  import FolderBrowser from '$lib/components/FolderBrowser.svelte';

  type ExportFormat = 'tif' | 'jpg' | 'png' | 'dng';
  let open = $state(false);
  let folder = $state('');
  let template = $state('{stack_name}_{method}');
  let format = $state<ExportFormat>('tif');
  let bitDepth = $state(16);
  let compression = $state('zlib');
  let jpegQuality = $state(95);
  let destOverride = $state<string | null>(null);
  let floatCompanion = $state(false);
  let depthCompanion = $state(false);
  let error = $state('');

  function resultId(): string | null {
    if (appState.viewer?.kind === 'result') return appState.viewer.id;
    return appState.results[appState.results.length - 1]?.id ?? null;
  }

  let result = $derived(appState.results.find((item) => item.id === resultId()));
  let rawResult = $derived(result?.domain === 'scene_linear_camera_rgb');
  let depthAvailable = $derived(
    !!appState.project && Object.values(appState.project.images).some(
      (image) => image.kind === 'depth' && image.parent === result?.id
    )
  );

  $effect(() => {
    const persisted = appState.project?.ui_state?.exportTemplate;
    if (typeof persisted === 'string' && persisted) template = persisted;
  });

  $effect(() => {
    if (format === 'jpg') bitDepth = 8;
    if (format === 'dng') bitDepth = 16;
  });

  function context(): TemplateCtx {
    const index = result ? appState.results.findIndex((item) => item.id === result.id) : 0;
    return {
      stack_name: appState.project?.name || 'stacked',
      method: result?.method ?? '',
      frames: result?.frames,
      date: todayISO(),
      seq: Math.max(index, 0) + 1
    };
  }

  let stem = $derived(applyTemplate(template, context()) || 'stacked');
  let filename = $derived(`${stem}.${format}`);
  let autoDest = $derived.by(() => {
    if (!folder) return '';
    const separator = folder.includes('\\') ? '\\' : '/';
    return `${folder.replace(/[\\/]+$/, '')}${separator}${filename}`;
  });
  let destination = $derived(destOverride ?? autoDest);

  function companionPath(suffix: string): string {
    if (!destination) return '';
    return destination.replace(/\.[^.\\/]+$/, '') + suffix;
  }

  let formatError = $derived.by(() => {
    if (!result) return 'Choose a finished result.';
    if (!destination) return 'Choose an output destination.';
    if (format === 'dng' && !rawResult) return 'Linear DNG is available only for a no-bake RAW result.';
    if (format === 'jpg' && bitDepth !== 8) return 'JPEG supports 8-bit output only.';
    if (!destination.toLowerCase().endsWith(`.${format}`)) return `Destination must end in .${format}.`;
    return '';
  });

  async function exportResult() {
    if (!appState.project || !result || formatError) return;
    error = '';
    try {
      const body: Record<string, unknown> = {
        image_id: result.id,
        dest: destination,
        format,
        bit_depth: bitDepth,
        compression,
        jpeg_quality: jpegQuality
      };
      if (floatCompanion) body.float_tiff_dest = companionPath('-scene-linear-float.tif');
      if (depthCompanion) body.depth_dest = companionPath('-depth.tif');
      const response = await api.export(appState.project.id, body);
      appState.exportJobId = response.id;
      void api.patchUiState(appState.project.id, { exportTemplate: template });
      open = false;
    } catch (cause) {
      error = (cause as Error).message;
    }
  }
</script>

<div class="export-wrap">
  <button
    class="ghost"
    data-testid="ws-export"
    disabled={!result}
    aria-expanded={open}
    onclick={() => (open = !open)}
    title={result ? 'Export selected result' : 'No result to export yet'}
  >Export{open ? ' ▲' : ' ▾'}</button>

  {#if open}
    <div class="popover panel" data-testid="export-panel">
      <div class="grid">
        <label>
          <span class="lbl">Format</span>
          <select bind:value={format} aria-label="Export format">
            <option value="tif">TIFF</option>
            <option value="png">PNG</option>
            <option value="jpg">JPEG</option>
            <option value="dng" disabled={!rawResult}>Linear DNG · Capture One</option>
          </select>
        </label>
        <label>
          <span class="lbl">Bit depth</span>
          <select bind:value={bitDepth} disabled={format === 'jpg' || format === 'dng'}>
            <option value={8}>8-bit</option>
            <option value={16}>16-bit</option>
          </select>
        </label>
        {#if format === 'tif'}
          <label>
            <span class="lbl">Compression</span>
            <select bind:value={compression}>
              <option value="none">None</option>
              <option value="lzw">LZW</option>
              <option value="zlib">ZIP</option>
            </select>
          </label>
        {/if}
        {#if format === 'jpg'}
          <label>
            <span class="lbl">Quality</span>
            <input type="number" bind:value={jpegQuality} min="1" max="100" />
          </label>
        {/if}
        <label class="wide">
          <span class="lbl">Name template</span>
          <input type="text" bind:value={template} data-testid="export-template" />
        </label>
        <div class="wide mono faint" data-testid="export-preview">{filename}</div>
        <div class="wide faint tokens">{`{stack_name} {method} {frames} {date} {seq}`}</div>
      </div>

      {#if format === 'dng'}
        <div class="no-bake" data-testid="dng-no-bake-summary">
          <strong>Maximum-information Linear DNG for current Capture One Pro</strong>
          <span>No baked white balance, gamma, auto brightness, tone curve, denoise, sharpening, or output-color conversion.</span>
          <span class="mono faint">16-bit lossless LinearRaw · LibRaw-validated · same-camera calibration and PlaneFuse provenance embedded</span>
        </div>
      {/if}

      <label class="check">
        <input type="checkbox" bind:checked={floatCompanion} />
        Also write an unclamped 32-bit float TIFF companion (not RAW)
      </label>
      <label class="check" title={!depthAvailable ? 'Available for DMap results with a depth map' : ''}>
        <input type="checkbox" bind:checked={depthCompanion} disabled={!depthAvailable} />
        Also export the 16-bit depth map
      </label>

      <div class="destination">
        <span class="lbl">Output folder</span>
        <FolderBrowser bind:value={folder} />
      </div>
      <label class="destination">
        <span class="lbl">Full path</span>
        <input
          class="mono"
          type="text"
          value={destination}
          data-testid="export-dest"
          oninput={(event) => (destOverride = event.currentTarget.value)}
          placeholder="/photos/stacked.tif"
        />
      </label>
      {#if formatError}<p class="validation-error">{formatError}</p>{/if}
      {#if error}<p class="err mono">{error}</p>{/if}
      <div class="actions">
        <button class="primary" data-testid="export-go" disabled={!!formatError} onclick={exportResult}>Export</button>
      </div>
    </div>
  {/if}
</div>

<style>
  .export-wrap { position: relative; }
  .popover { position: absolute; right: 0; top: calc(100% + 8px); z-index: 30; width: min(560px, 88vw); padding: 14px; box-shadow: 0 18px 50px rgba(0,0,0,.45); }
  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 9px; }
  label { display: flex; flex-direction: column; gap: 4px; }
  .wide { grid-column: 1 / -1; }
  .tokens { font-size: 10px; }
  .no-bake { display: flex; flex-direction: column; gap: 5px; margin: 10px 0; padding: 10px; border: 1px solid var(--accent); border-radius: 4px; background: var(--accent-dim); font-size: 10px; line-height: 1.35; }
  .check { flex-direction: row; align-items: center; margin-top: 8px; font-size: 11px; }
  .destination { margin-top: 10px; }
  .destination input { width: 100%; }
  .actions { display: flex; justify-content: flex-end; margin-top: 12px; }
  .validation-error, .err { color: var(--bad); font-size: 10px; margin: 8px 0 0; }
  .lbl { color: var(--text-dim); font-size: 10px; }
</style>
