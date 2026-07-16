<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import { api } from '$lib/api';
  import { appState } from '$lib/stores.svelte';
  import { isEditableTarget } from '$lib/shortcuts';

  let {
    imageId,
    levels,
    width,
    height,
    onmove,
    tileVersion,
    brushRadius,
    onstroke,
    view
  }: {
    imageId: string;
    levels: number;
    width: number;
    height: number;
    onmove?: (v: { scale: number; tx: number; ty: number }) => void;
    tileVersion?: (z: number, x: number, y: number) => number;
    brushRadius?: number;
    onstroke?: (points: [number, number, number][]) => void;
    view?: { scale: number; tx: number; ty: number } | null;
  } = $props();

  const TILE = 256;

  let viewport: HTMLDivElement;
  let cw = $state(0);
  let ch = $state(0);

  // scale = screen px per full-res image px; (tx,ty) = screen pos of image (0,0).
  let scale = $state(1);
  let tx = $state(0);
  let ty = $state(0);

  // Sync FROM an external `view` (e.g. the other pane in a synchronized
  // compare view) without this effect also depending on our OWN scale/tx/ty.
  // Reading them via `untrack` keeps the effect's only dependency on `view`
  // itself — its reference only changes when a *parent* re-render passes a
  // new one, never merely because a local zoom/pan/fit here just wrote to
  // `scale`. Without `untrack`, this effect re-fires on every local write to
  // `scale` too; since `emit()`'s round trip through `onmove` back to `view`
  // is deferred a frame (requestAnimationFrame), it would read back the
  // *stale* pre-change `view` and immediately revert the local change —
  // every zoom, pan, or fit silently undone the instant it happens.
  $effect(() => {
    if (!view) return;
    const next = view;
    untrack(() => {
      if (Math.abs(scale - next.scale) > 1e-6) scale = next.scale;
      if (Math.abs(tx - next.tx) > 1e-6) tx = next.tx;
      if (Math.abs(ty - next.ty) > 1e-6) ty = next.ty;
    });
  });

  function fit() {
    if (!cw || !ch) return;
    scale = Math.min(cw / width, ch / height);
    tx = (cw - width * scale) / 2;
    ty = (ch - height * scale) / 2;
    emit();
  }

  function actualSize() {
    const mx = cw / 2;
    const my = ch / 2;
    zoomAround(mx, my, 1 / scale);
  }

  function zoomAround(px: number, py: number, factor: number) {
    const ns = Math.max(0.02, Math.min(scale * factor, 16));
    // keep the image point under (px,py) fixed
    tx = px - ((px - tx) * ns) / scale;
    ty = py - ((py - ty) * ns) / scale;
    scale = ns;
    emit();
  }

  function onwheel(e: WheelEvent) {
    e.preventDefault();
    const r = viewport.getBoundingClientRect();
    zoomAround(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.15 : 1 / 1.15);
  }

  let dragging = false;
  let painting = false;
  let spaceHeld = $state(false);
  let strokePoints: [number, number, number][] = [];
  let lastX = 0;
  let lastY = 0;
  let pointerX = $state(0);
  let pointerY = $state(0);
  let pointerInside = $state(false);

  function point(e: PointerEvent): [number, number, number] | null {
    const r = viewport.getBoundingClientRect();
    pointerX = e.clientX - r.left;
    pointerY = e.clientY - r.top;
    const x = (pointerX - tx) / scale;
    const y = (pointerY - ty) / scale;
    if (x < 0 || y < 0 || x >= width || y >= height) return null;
    return [x, y, e.pressure > 0 ? e.pressure : 1];
  }

  function ondown(e: PointerEvent) {
    if (e.button !== 0) return;
    viewport.focus();
    viewport.setPointerCapture(e.pointerId);
    if (onstroke && !spaceHeld) {
      painting = true;
      strokePoints = [];
      const p = point(e);
      if (p) strokePoints.push(p);
    } else {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
    }
  }
  function onmovep(e: PointerEvent) {
    const p = point(e);
    if (painting) {
      if (p) strokePoints.push(p);
      return;
    }
    if (!dragging) return;
    tx += e.clientX - lastX;
    ty += e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    emit();
  }
  function onup(e: PointerEvent) {
    if (painting) {
      const p = point(e);
      if (p) strokePoints.push(p);
      painting = false;
      if (strokePoints.length) onstroke?.(strokePoints);
      strokePoints = [];
    }
    dragging = false;
    if (viewport.hasPointerCapture(e.pointerId)) viewport.releasePointerCapture(e.pointerId);
  }

  function onkey(e: KeyboardEvent) {
    if (isEditableTarget(e.target)) return;
    if (e.code === 'Space' && onstroke) {
      spaceHeld = true;
      e.preventDefault();
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === 'f' || e.key === 'F') fit();
    else if (e.key === 'z' || e.key === 'Z') actualSize();
  }

  function onkeyup(e: KeyboardEvent) {
    if (e.code === 'Space') spaceHeld = false;
  }

  let raf = 0;
  function emit() {
    if (!onmove) return;
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => onmove?.({ scale, tx, ty }));
  }

  // Choose the pyramid level whose native resolution just covers the display.
  let z = $derived(
    Math.max(0, Math.min(levels - 1, Math.ceil(levels - 1 + Math.log2(scale))))
  );
  let f = $derived(2 ** (levels - 1 - z)); // full-res px per level px
  let lw = $derived(Math.max(1, Math.floor(width / f)));
  let lh = $derived(Math.max(1, Math.floor(height / f)));
  let cols = $derived(Math.ceil(lw / TILE));
  let rows = $derived(Math.ceil(lh / TILE));
  let fullTile = $derived(TILE * f * scale); // screen px for a full 256px tile

  // Visible tiles (cull off-screen). Edge tiles are smaller than 256px, so each
  // tile carries its own clamped width/height to avoid stretching.
  let visible = $derived.by(() => {
    const out: { i: number; j: number; left: number; top: number; w: number; h: number }[] = [];
    if (!cw || !ch) return out;
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const left = tx + i * fullTile;
        const top = ty + j * fullTile;
        const tw = (Math.min((i + 1) * TILE, lw) - i * TILE) * f * scale;
        const th = (Math.min((j + 1) * TILE, lh) - j * TILE) * f * scale;
        if (left + tw < 0 || top + th < 0 || left > cw || top > ch) continue;
        out.push({ i, j, left, top, w: tw, h: th });
      }
    }
    return out;
  });

  onMount(() => {
    const ro = new ResizeObserver(() => {
      cw = viewport.clientWidth;
      ch = viewport.clientHeight;
      if (scale === 1 && tx === 0 && ty === 0) fit();
    });
    ro.observe(viewport);
    return () => ro.disconnect();
  });

  export function reset() {
    fit();
  }
</script>

<svelte:window onkeydown={onkey} onkeyup={onkeyup} />

<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div
  class="viewport"
  data-testid="deep-zoom"
  bind:this={viewport}
  role="application"
  aria-label="Deep-zoom image viewer"
  tabindex="0"
  class:editing={!!onstroke && !spaceHeld}
  onwheel={onwheel}
  onpointerdown={ondown}
  onpointermove={onmovep}
  onpointerup={onup}
  onpointercancel={onup}
  onpointerenter={() => (pointerInside = true)}
  onpointerleave={() => (pointerInside = false)}
>
  {#each visible as t (z + '-' + t.i + '-' + t.j)}
    <img
      class="tile"
      data-testid="viewer-tile"
      src={api.tileUrl(imageId, z, t.i, t.j, tileVersion?.(z, t.i, t.j), appState.displayTonemap)}
      alt=""
      draggable="false"
      style:left="{t.left}px"
      style:top="{t.top}px"
      style:width="{t.w}px"
      style:height="{t.h}px"
    />
  {/each}

  {#if onstroke && brushRadius && pointerInside && !spaceHeld}
    <div
      class="brush"
      style:left="{pointerX - brushRadius * scale}px"
      style:top="{pointerY - brushRadius * scale}px"
      style:width="{brushRadius * scale * 2}px"
      style:height="{brushRadius * scale * 2}px"
    ></div>
  {/if}

  <div class="hud mono">
    <span>{Math.round(scale * 100)}%</span>
    <span class="faint">z{z}</span>
    <span class="faint">·</span>
    <span class="faint"><span class="kbd">F</span> fit <span class="kbd">Z</span> 100%</span>
  </div>
</div>

<style>
  .viewport {
    position: relative;
    width: 100%;
    height: 100%;
    overflow: hidden;
    touch-action: none;
    background:
      repeating-conic-gradient(#0d0e11 0% 25%, #0a0b0e 0% 50%) 50% / 24px 24px;
    cursor: grab;
    outline: none;
  }
  .viewport:active { cursor: grabbing; }
  .viewport.editing { cursor: none; }
  .tile {
    position: absolute;
    image-rendering: auto;
    user-select: none;
    -webkit-user-drag: none;
  }
  .brush {
    position: absolute;
    z-index: 2;
    border: 1px solid rgba(255, 255, 255, 0.9);
    border-radius: 50%;
    box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.75);
    pointer-events: none;
  }
  .hud {
    position: absolute;
    left: 12px;
    bottom: 12px;
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 5px 10px;
    font-size: 12px;
    background: rgba(10, 11, 14, 0.8);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    backdrop-filter: blur(4px);
  }
</style>
