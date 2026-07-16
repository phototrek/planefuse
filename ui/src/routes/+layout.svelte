<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { api } from '$lib/api';
  import { appState } from '$lib/stores.svelte';
  import { connectJobs } from '$lib/ws';

  let { children } = $props();

  let memGiB = $derived(
    appState.system ? (appState.system.free_memory / 1024 ** 3).toFixed(1) : '—'
  );

  onMount(() => {
    void api.system().then((system) => {
      appState.system = system;
    }).catch(() => {
      /* banner shows offline */
    });
    const refreshJobs = async () => {
      try {
        const jobs = await api.listJobs();
        for (const job of jobs) appState.jobs[job.id] = job;
      } catch {
        /* WebSocket reconnect and next poll will recover. */
      }
    };
    void refreshJobs();
    const poll = setInterval(refreshJobs, 750);
    connectJobs();
    return () => clearInterval(poll);
  });
</script>

<div id="app-grain"></div>

<div class="app">
  <header class="topbar">
    <a class="brand" href="/">
      <span class="bracket">⌖</span>
      <span class="word">Plane<span class="accent">Fuse</span></span>
    </a>

    <div class="project">
      {#if appState.project}
        <span class="eyebrow">project</span>
        <span class="pname">{appState.project.name}</span>
      {:else}
        <span class="faint mono">no project open</span>
      {/if}
    </div>

    <div class="device mono" title="Compute device and free memory (GET /api/system)">
      {#if appState.system}
        <span class="dot" class:cpu={appState.system.device === 'cpu'}></span>
        <span class="dev">{appState.system.device.toUpperCase()}</span>
        <span class="faint">·</span>
        <span class="dim">{memGiB} GiB free</span>
      {:else}
        <span class="dot off"></span><span class="faint">connecting…</span>
      {/if}
    </div>

    <a class="help-link" href="/help" title="Documentation and troubleshooting">Help</a>
  </header>

  <main class="main">
    {@render children()}
  </main>
</div>

<style>
  .app { display: grid; grid-template-rows: var(--topbar-h) 1fr; height: 100vh; }

  .topbar {
    display: grid;
    grid-template-columns: auto 1fr auto auto;
    align-items: center;
    border-bottom: 1px solid var(--line);
    background: linear-gradient(180deg, var(--panel-2), var(--panel));
  }

  .help-link {
    padding: 0 18px;
    height: 100%;
    display: flex;
    align-items: center;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-dim);
    border-left: 1px solid var(--line);
  }
  .help-link:hover { color: var(--accent-bright); }

  .main { overflow: hidden; }   /* workspace manages its own scrolling */

  .brand {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 0 18px;
    height: 100%;
    border-right: 1px solid var(--line);
  }
  .brand .bracket { color: var(--accent); font-size: 18px; line-height: 1; }
  .brand .word { font-weight: 700; letter-spacing: -0.02em; font-size: 16px; }
  .brand .accent { color: var(--accent); }

  .project { display: flex; align-items: baseline; gap: 10px; padding: 0 20px; min-width: 0; }
  .project .pname { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  .device {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 18px;
    font-size: 12px;
  }
  .device .dev { color: var(--accent-bright); font-weight: 500; }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--good); box-shadow: 0 0 8px var(--good); }
  .dot.cpu { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
  .dot.off { background: var(--text-faint); box-shadow: none; }
</style>
