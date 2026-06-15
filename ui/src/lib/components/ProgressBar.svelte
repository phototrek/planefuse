<script lang="ts">
  let { percent = 0, status = 'running' }: { percent?: number; status?: string } = $props();
  let pct = $derived(Math.round(Math.max(0, Math.min(1, percent)) * 100));
</script>

<div class="bar" class:done={status === 'done'} class:error={status === 'error'} class:cancelled={status === 'cancelled'}>
  <div class="fill" style:width="{pct}%"></div>
  <span class="pct mono">{pct}%</span>
</div>

<style>
  .bar {
    position: relative;
    height: 22px;
    background: var(--bg);
    border: 1px solid var(--line-strong);
    border-radius: var(--radius);
    overflow: hidden;
  }
  .fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent-bright));
    transition: width 0.25s ease;
  }
  .done .fill { background: var(--good); }
  .error .fill { background: var(--bad); }
  .cancelled .fill { background: var(--text-faint); }
  .pct {
    position: absolute;
    top: 0;
    right: 8px;
    line-height: 22px;
    font-size: 11px;
    color: var(--text);
    mix-blend-mode: difference;
  }
</style>
