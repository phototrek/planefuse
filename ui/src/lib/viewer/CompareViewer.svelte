<script lang="ts">
  import DeepZoom from './DeepZoom.svelte';
  import type { CompareMode, ViewerTransform } from '$lib/stores.svelte';

  interface View { imageId: string; levels: number; width: number; height: number; }
  let {
    primary,
    secondary,
    mode,
    transform,
    onmove
  }: {
    primary: View;
    secondary: View;
    mode: CompareMode;
    transform: ViewerTransform | null;
    onmove: (view: ViewerTransform) => void;
  } = $props();

  let divider = $state(50);
  let showBefore = $state(false);

  function keydown(event: KeyboardEvent) {
    if (event.key === '\\') {
      showBefore = true;
      event.preventDefault();
    }
  }
  function keyup(event: KeyboardEvent) {
    if (event.key === '\\') showBefore = false;
  }
</script>

<svelte:window onkeydown={keydown} onkeyup={keyup} />

<div class="compare" data-testid="compare-viewer">
  {#if mode === 'side-by-side'}
    <div class="side"><DeepZoom {...primary} view={transform} {onmove} /></div>
    <div class="side"><DeepZoom {...secondary} view={transform} {onmove} /></div>
  {:else if mode === 'before-after'}
    {#if showBefore}
      <DeepZoom {...secondary} view={transform} {onmove} />
    {:else}
      <DeepZoom {...primary} view={transform} {onmove} />
    {/if}
    <button
      class="hold"
      onpointerdown={() => (showBefore = true)}
      onpointerup={() => (showBefore = false)}
      onpointerleave={() => (showBefore = false)}
    >Hold for before <span class="kbd">\</span></button>
  {:else}
    <div class="layer"><DeepZoom {...primary} view={transform} {onmove} /></div>
    <div class="layer clipped" style:clip-path={`inset(0 ${100 - divider}% 0 0)`}>
      <DeepZoom {...secondary} view={transform} {onmove} />
    </div>
    <input
      class="divider"
      type="range"
      min="0"
      max="100"
      bind:value={divider}
      aria-label="Comparison divider"
    />
    <div class="divider-line" style:left={`${divider}%`}></div>
  {/if}
</div>

<style>
  .compare { position: absolute; inset: 0; display: grid; grid-template-columns: 1fr 1fr; overflow: hidden; }
  .side { position: relative; min-width: 0; border-right: 1px solid var(--line); }
  .layer { position: absolute; inset: 0; }
  .clipped { z-index: 2; }
  .divider { position: absolute; inset: 0; z-index: 5; width: 100%; height: 100%; opacity: 0; cursor: ew-resize; }
  .divider-line { position: absolute; top: 0; bottom: 0; width: 1px; z-index: 4; background: rgba(255,255,255,.8); box-shadow: 0 0 0 1px rgba(0,0,0,.7); pointer-events: none; }
  .hold { position: absolute; z-index: 6; bottom: 12px; right: 12px; }
</style>
