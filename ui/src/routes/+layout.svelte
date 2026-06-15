<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { api } from '$lib/api';
  import { appState } from '$lib/stores.svelte';
  import { connectJobs } from '$lib/ws';

  let { children } = $props();

  const NAV = [
    { href: '/', label: 'Import', key: '1', icon: 'folder' },
    { href: '/stack', label: 'Stack', key: '2', icon: 'layers' },
    { href: '/queue', label: 'Queue', key: '3', icon: 'pulse' },
    { href: '/viewer', label: 'Viewer', key: '4', icon: 'eye' },
    { href: '/export', label: 'Export', key: '5', icon: 'export' }
  ];

  let memGiB = $derived(
    appState.system ? (appState.system.free_memory / 1024 ** 3).toFixed(1) : '—'
  );

  onMount(async () => {
    try {
      appState.system = await api.system();
    } catch {
      /* banner shows offline */
    }
    connectJobs();
  });
</script>

<div id="app-grain"></div>

<div class="app">
  <header class="topbar">
    <a class="brand" href="/">
      <span class="bracket">⌖</span>
      <span class="word">Focus<span class="accent">Stack</span></span>
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
  </header>

  <nav class="rail">
    {#each NAV as item (item.href)}
      <a
        class="navlink"
        class:active={$page.url.pathname === item.href}
        href={item.href}
        title="{item.label} (press {item.key})"
        data-testid="nav-{item.label.toLowerCase()}"
      >
        <span class="ico ico-{item.icon}" aria-hidden="true"></span>
        <span class="nlabel">{item.label}</span>
        <span class="kbd nkey">{item.key}</span>
      </a>
    {/each}
    <div class="rail-foot mono faint">v{appState.system?.version ?? '0.1.0'}</div>
  </nav>

  <main class="main">
    {@render children()}
  </main>
</div>

<style>
  .app {
    display: grid;
    grid-template-columns: var(--rail-w) 1fr;
    grid-template-rows: var(--topbar-h) 1fr;
    height: 100vh;
  }

  .topbar {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: var(--rail-w) 1fr auto;
    align-items: center;
    border-bottom: 1px solid var(--line);
    background: linear-gradient(180deg, var(--panel-2), var(--panel));
  }

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

  .rail {
    grid-row: 2;
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 12px 10px;
    border-right: 1px solid var(--line);
    background: var(--panel);
  }

  .navlink {
    display: grid;
    grid-template-columns: 20px 1fr auto;
    align-items: center;
    gap: 12px;
    padding: 9px 11px;
    border-radius: var(--radius);
    color: var(--text-dim);
    border: 1px solid transparent;
    transition: background 0.12s, color 0.12s, border-color 0.12s;
  }
  .navlink:hover { background: var(--panel-2); color: var(--text); }
  .navlink.active {
    color: var(--text);
    background: var(--accent-dim);
    border-color: var(--accent-line);
  }
  .navlink.active .ico { background: var(--accent); }
  .nlabel { font-weight: 500; }
  .nkey { opacity: 0.6; }

  /* simple geometric icons via masks */
  .ico {
    width: 18px; height: 18px;
    background: var(--text-dim);
    -webkit-mask-position: center; mask-position: center;
    -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
    -webkit-mask-size: contain; mask-size: contain;
  }
  .ico-folder { -webkit-mask-image: var(--m-folder); mask-image: var(--m-folder); }
  .ico-layers { -webkit-mask-image: var(--m-layers); mask-image: var(--m-layers); }
  .ico-pulse { -webkit-mask-image: var(--m-pulse); mask-image: var(--m-pulse); }
  .ico-eye { -webkit-mask-image: var(--m-eye); mask-image: var(--m-eye); }
  .ico-export { -webkit-mask-image: var(--m-export); mask-image: var(--m-export); }

  .rail {
    --m-folder: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2'%3E%3Cpath d='M3 6h6l2 2h10v11H3z'/%3E%3C/svg%3E");
    --m-layers: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2'%3E%3Cpath d='M12 3 2 8l10 5 10-5z'/%3E%3Cpath d='M2 14l10 5 10-5'/%3E%3C/svg%3E");
    --m-pulse: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2'%3E%3Cpath d='M2 12h5l3 8 4-16 3 8h5'/%3E%3C/svg%3E");
    --m-eye: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2'%3E%3Cpath d='M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12z'/%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3C/svg%3E");
    --m-export: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2'%3E%3Cpath d='M12 3v12'/%3E%3Cpath d='M7 8l5-5 5 5'/%3E%3Cpath d='M5 21h14'/%3E%3C/svg%3E");
  }

  .rail-foot { margin-top: auto; padding: 8px 11px; font-size: 11px; }

  .main { grid-row: 2; overflow: auto; }
</style>
