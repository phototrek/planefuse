<script lang="ts">
  import { appState } from '$lib/stores.svelte';

  const RAW_DOMAIN = 'scene_linear_camera_rgb';
  let report = $derived(appState.scanReport);
  let blockers = $derived(report?.files.filter((file) => file.status !== 'ok') ?? []);
  let isRaw = $derived(report?.domain === RAW_DOMAIN);

  function label(status: string): string {
    return ({
      wrong_size: 'Different dimensions',
      wrong_bit_depth: 'Different bit depth',
      mixed_domain: 'Mixed RAW and rendered files',
      incompatible_camera: 'Different camera',
      incompatible_sensor_mode: 'Different sensor mode',
      incompatible_cfa: 'Different sensor pattern',
      incompatible_raw_calibration: 'Different camera calibration',
      missing_raw_calibration: 'Missing RAW calibration',
      raw_decode_error: 'RAW could not be decoded',
      unreadable: 'Unreadable file',
      unsupported: 'Unsupported file'
    } as Record<string, string>)[status] ?? status.replaceAll('_', ' ');
  }
</script>

{#if report && report.files.length > 0}
  <section class="validation" class:valid={report.ok} aria-live="polite" data-testid="validation-panel">
    <div class="summary-row">
      <strong>{report.ok ? 'Ready to stack' : `${blockers.length} blocking issue${blockers.length === 1 ? '' : 's'}`}</strong>
      <span class="mono faint">
        {report.files.length} frames
        {#if report.width && report.height} · {report.width}×{report.height}{/if}
        {#if report.bit_depth} · {report.bit_depth}-bit{/if}
      </span>
    </div>

    {#if isRaw}
      <div class="raw-contract" data-testid="raw-mode-summary">
        <span class="raw-pill">RAW · same camera</span>
        <strong>{report.camera || 'Camera metadata pending'}</strong>
        <p>No aesthetic development is baked: no white balance, gamma, auto brightness, tone curve, denoise, sharpening, or output-color conversion.</p>
        <span class="mono faint">
          {String(report.decoder.demosaic ?? 'AHD')} demosaic · {String(report.decoder.library ?? 'LibRaw')} · scene-linear camera RGB
        </span>
      </div>
    {/if}

    {#if blockers.length > 0}
      <ul class="issues">
        {#each blockers as blocker (blocker.path)}
          <li>
            <a href={'#input-' + encodeURIComponent(blocker.path)}>{blocker.name}</a>
            <span>{label(blocker.status)}</span>
            {#if blocker.message}<small class="mono">{blocker.message}</small>{/if}
          </li>
        {/each}
      </ul>
    {/if}

    {#if report.files.length > 40}
      <p class="recommendation">Large stack: smart frame selection is recommended. Review the proposal before interactive stacking.</p>
    {/if}
  </section>
{/if}

<style>
  .validation { margin: 6px 8px; padding: 9px; border: 1px solid var(--bad); border-radius: var(--radius); background: color-mix(in srgb, var(--bad) 8%, var(--panel)); }
  .validation.valid { border-color: var(--good); background: color-mix(in srgb, var(--good) 6%, var(--panel)); }
  .summary-row { display: flex; flex-direction: column; gap: 2px; font-size: 11px; }
  .raw-contract { display: flex; flex-direction: column; gap: 5px; margin-top: 8px; padding: 8px; border-radius: 4px; background: var(--panel-2); font-size: 10px; }
  .raw-contract p { margin: 0; line-height: 1.35; color: var(--text-dim); }
  .raw-pill { align-self: flex-start; color: var(--accent-bright); font-family: var(--font-mono); letter-spacing: 0.05em; }
  .issues { list-style: none; margin: 8px 0 0; padding: 0; display: flex; flex-direction: column; gap: 7px; }
  .issues li { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 2px 6px; font-size: 10px; }
  .issues a { color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .issues span { color: var(--bad); }
  .issues small { grid-column: 1 / -1; color: var(--text-faint); overflow-wrap: anywhere; }
  .recommendation { margin: 8px 0 0; color: var(--warn, #e1a857); font-size: 10px; line-height: 1.35; }
</style>
