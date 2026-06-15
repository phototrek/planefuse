<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/api';

  let {
    imageId,
    levels,
    width,
    height,
    onmove
  }: {
    imageId: string;
    levels: number;
    width: number;
    height: number;
    onmove?: (v: { scale: number; tx: number; ty: number }) => void;
  } = $props();

  const TILE = 256;

  let viewport: HTMLDivElement;
  let cw = $state(0);
  let ch = $state(0);

  // scale = screen px per full-res image px; (tx,ty) = screen pos of image (0,0).
  let scale = $state(1);
  let tx = $state(0);
  let ty = $state(0);

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
  let lastX = 0;
  let lastY = 0;
  function ondown(e: PointerEvent) {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    viewport.setPointerCapture(e.pointerId);
  }
  function onmovep(e: PointerEvent) {
    if (!dragging) return;
    tx += e.clientX - lastX;
    ty += e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    emit();
  }
  function onup(e: PointerEvent) {
    dragging = false;
    viewport.releasePointerCapture(e.pointerId);
  }

  function onkey(e: KeyboardEvent) {
    if (e.key === 'f' || e.key === 'F') fit();
    else if (e.key === 'z' || e.key === 'Z') actualSize();
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

<svelte:window onkeydown={onkey} />

<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div
  class="viewport"
  bind:this={viewport}
  role="application"
  aria-label="Deep-zoom image viewer"
  tabindex="0"
  onwheel={onwheel}
  onpointerdown={ondown}
  onpointermove={onmovep}
  onpointerup={onup}
>
  {#each visible as t (z + '-' + t.i + '-' + t.j)}
    <img
      class="tile"
      data-testid="viewer-tile"
      src={api.tileUrl(imageId, z, t.i, t.j)}
      alt=""
      draggable="false"
      style:left="{t.left}px"
      style:top="{t.top}px"
      style:width="{t.w}px"
      style:height="{t.h}px"
    />
  {/each}

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
    background:
      repeating-conic-gradient(#0d0e11 0% 25%, #0a0b0e 0% 50%) 50% / 24px 24px;
    cursor: grab;
    outline: none;
  }
  .viewport:active { cursor: grabbing; }
  .tile {
    position: absolute;
    image-rendering: auto;
    user-select: none;
    -webkit-user-drag: none;
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
