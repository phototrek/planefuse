<script lang="ts">
  import type { ImageAnalysis } from '$lib/api';

  let { analysis }: { analysis: ImageAnalysis } = $props();
  const colors = {
    red: '#ff5c68',
    green: '#58d68d',
    blue: '#62a8ff',
    luminance: '#e9edf4'
  } as const;

  function points(values: number[]): string {
    const transformed = values.map((value) => Math.log1p(value));
    const maximum = Math.max(...transformed, 1);
    return transformed
      .map((value, index) => `${(index / 255) * 100},${36 - (value / maximum) * 34}`)
      .join(' ');
  }

  function clipped(channel: 'red' | 'green' | 'blue'): number {
    return analysis.clipping.shadows[channel] + analysis.clipping.highlights[channel];
  }
</script>

<section class="histogram panel" data-testid="histogram" aria-label="Image histogram">
  <svg viewBox="0 0 100 38" preserveAspectRatio="none" role="img" aria-label="RGB and luminance histogram">
    {#each Object.entries(analysis.histograms) as [name, bins]}
      <polyline
        points={points(bins)}
        fill="none"
        stroke={colors[name as keyof typeof colors]}
        stroke-width={name === 'luminance' ? 0.55 : 0.4}
        opacity={name === 'luminance' ? 0.8 : 0.7}
        vector-effect="non-scaling-stroke"
      />
    {/each}
  </svg>
  <div class="clip-row" aria-label="Clipping indicators">
    {#each ['red', 'green', 'blue'] as channel}
      <span class="clip" class:active={clipped(channel as 'red' | 'green' | 'blue') > 0} style:--clip-color={colors[channel as keyof typeof colors]}>
        {channel[0].toUpperCase()} {analysis.clipping.shadows[channel as 'red' | 'green' | 'blue']}↓ / {analysis.clipping.highlights[channel as 'red' | 'green' | 'blue']}↑
      </span>
    {/each}
  </div>
</section>

<style>
  .histogram { position: absolute; top: 50px; right: 12px; z-index: 8; width: 240px; padding: 8px; background: color-mix(in srgb, var(--panel) 90%, transparent); backdrop-filter: blur(8px); }
  svg { display: block; width: 100%; height: 86px; background: #090a0d; border-radius: 3px; }
  .clip-row { display: flex; gap: 7px; margin-top: 6px; font-family: var(--font-mono); font-size: 9px; color: var(--text-faint); }
  .clip.active { color: var(--clip-color); }
</style>
