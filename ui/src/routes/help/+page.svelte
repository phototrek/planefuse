<script lang="ts">
  import { onMount } from 'svelte';
  import { docGroups, docSections, type DocSection } from '$lib/help/content';

  let query = $state('');
  let activeId = $state('');

  // Strip tags once per section so search can match body text, not just the title.
  const plainText = new Map<string, string>(
    docSections.map((s) => [s.id, `${s.title} ${(s.keywords ?? []).join(' ')} ${s.html.replace(/<[^>]+>/g, ' ')}`.toLowerCase()])
  );

  let matches = $derived.by(() => {
    const q = query.trim().toLowerCase();
    if (!q) return new Set(docSections.map((s) => s.id));
    return new Set(docSections.filter((s) => plainText.get(s.id)?.includes(q)).map((s) => s.id));
  });

  let groups = $derived(
    docGroups
      .map((group) => ({ group, sections: docSections.filter((s) => s.group === group && matches.has(s.id)) }))
      .filter((g) => g.sections.length > 0)
  );

  let visibleSections = $derived(docSections.filter((s) => matches.has(s.id)));

  function jump(id: string, event?: MouseEvent) {
    event?.preventDefault();
    activeId = id;
    location.hash = id;
    document.getElementById(id)?.scrollIntoView({ block: 'start' });
  }

  onMount(() => {
    const hash = location.hash.replace('#', '');
    if (hash) {
      activeId = hash;
      // Instant, not smooth: .content has scroll-behavior: smooth for in-app
      // nav clicks, but stacking several *animated* scrollIntoView calls here
      // (each restarting the in-flight animation) leaves the final position
      // wherever the last interrupted animation happened to be. Re-settling
      // instantly after fonts/layout finish is idempotent instead.
      const scrollToHash = () => document.getElementById(hash)?.scrollIntoView({ block: 'start', behavior: 'instant' });
      requestAnimationFrame(scrollToHash);
      document.fonts?.ready?.then(scrollToHash);
      setTimeout(scrollToHash, 300);
    }

    const sections = docSections.map((s) => document.getElementById(s.id)).filter((el): el is HTMLElement => !!el);
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) activeId = visible[0].target.id;
      },
      { rootMargin: '-15% 0px -70% 0px' }
    );
    for (const el of sections) observer.observe(el);
    return () => observer.disconnect();
  });
</script>

<svelte:head>
  <title>Help — PlaneFuse</title>
</svelte:head>

<div class="help">
  <nav class="sidebar panel" aria-label="Documentation sections">
    <div class="search-row">
      <input
        type="search"
        placeholder="Search the docs…"
        bind:value={query}
        aria-label="Search documentation"
      />
    </div>
    {#if groups.length === 0}
      <p class="no-results faint">No sections match "{query}".</p>
    {/if}
    {#each groups as g (g.group)}
      <div class="nav-group">
        <p class="nav-group-title">{g.group}</p>
        <ul>
          {#each g.sections as s (s.id)}
            <li>
              <a href={'#' + s.id} class:active={activeId === s.id} onclick={(e) => jump(s.id, e)}>{s.title}</a>
            </li>
          {/each}
        </ul>
      </div>
    {/each}
    <div class="back-row">
      <a class="back-link" href="/">← Back to workspace</a>
    </div>
  </nav>

  <main class="content">
    <header class="content-header">
      <p class="eyebrow">PlaneFuse documentation</p>
      <h1>Help</h1>
      <p class="lede">Everything you need to shoot, stack, review, retouch, and export — written for
        photographers, with the technical detail included rather than hidden.</p>
    </header>

    {#if visibleSections.length === 0}
      <p class="faint">No sections match your search.</p>
    {/if}

    {#each docGroups as group}
      {@const sections = visibleSections.filter((s) => s.group === group)}
      {#if sections.length > 0}
        <section class="doc-group">
          <h2 class="group-heading">{group}</h2>
          {#each sections as s (s.id)}
            <article class="doc-section" id={s.id}>
              <h2 class="section-title"><a href={'#' + s.id} onclick={(e) => jump(s.id, e)}>{s.title}</a></h2>
              <div class="doc-body">{@html s.html}</div>
            </article>
          {/each}
        </section>
      {/if}
    {/each}
  </main>
</div>

<style>
  .help {
    display: grid;
    grid-template-columns: 260px 1fr;
    height: 100%;
    overflow: hidden;
  }

  /* ── Sidebar ── */
  .sidebar {
    border-right: 1px solid var(--line);
    border-radius: 0;
    overflow-y: auto;
    padding: 14px 10px 20px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .search-row { position: sticky; top: 0; background: var(--panel); padding-bottom: 10px; z-index: 2; }
  .search-row input {
    width: 100%;
    font-size: 13px;
    padding: 8px 10px;
  }
  .no-results { padding: 10px 6px; font-size: 12px; }
  .nav-group { margin-top: 10px; }
  .nav-group-title {
    margin: 0 0 4px;
    padding: 0 8px;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-faint);
  }
  .nav-group ul { list-style: none; margin: 0; padding: 0; }
  .nav-group a {
    display: block;
    padding: 6px 8px;
    border-radius: var(--radius);
    font-size: 12.5px;
    color: var(--text-dim);
    border-left: 2px solid transparent;
    line-height: 1.3;
  }
  .nav-group a:hover { color: var(--text); background: var(--panel-2); }
  .nav-group a.active {
    color: var(--accent-bright);
    background: var(--accent-dim);
    border-left-color: var(--accent);
    font-weight: 500;
  }
  .back-row { margin-top: auto; padding: 14px 8px 4px; border-top: 1px solid var(--line); }
  .back-link { font-size: 12px; color: var(--text-faint); }
  .back-link:hover { color: var(--accent-bright); }

  /* ── Content ── */
  .content {
    overflow-y: auto;
    padding: 32px clamp(20px, 5vw, 64px) 120px;
    scroll-behavior: smooth;
  }
  .content-header { max-width: 720px; margin-bottom: 36px; }
  .content-header h1 { font-size: 30px; margin: 4px 0 10px; }
  .content-header .lede { color: var(--text-dim); font-size: 15px; line-height: 1.5; margin: 0; }
  .eyebrow { font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent); }

  .doc-group { max-width: 720px; margin-bottom: 40px; }
  .group-heading {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-faint);
    padding-bottom: 8px;
    margin-bottom: 22px;
    border-bottom: 1px solid var(--line);
  }
  .doc-section { scroll-margin-top: 18px; margin-bottom: 34px; }
  .section-title { font-size: 19px; margin-bottom: 12px; }
  .section-title a { color: var(--text); }
  .section-title a:hover { color: var(--accent-bright); }

  /* ── Rendered doc body (shared typography for {@html} content) ── */
  .doc-body { color: var(--text-dim); font-size: 14px; line-height: 1.65; }
  .doc-body :global(p) { margin: 0 0 12px; }
  .doc-body :global(h3) { color: var(--text); font-size: 15px; margin: 20px 0 8px; }
  .doc-body :global(strong) { color: var(--text); font-weight: 600; }
  .doc-body :global(em) { color: var(--text); }
  .doc-body :global(a) { color: var(--accent-bright); text-decoration: underline; text-decoration-color: var(--accent-line); text-underline-offset: 2px; }
  .doc-body :global(ul), .doc-body :global(ol) { margin: 0 0 14px; padding-left: 22px; display: flex; flex-direction: column; gap: 6px; }
  .doc-body :global(li) { padding-left: 2px; }
  .doc-body :global(li strong) { color: var(--text); }
  .doc-body :global(code) {
    font-family: var(--font-mono);
    font-size: 12.5px;
    background: var(--panel-3);
    border: 1px solid var(--line);
    border-radius: 3px;
    padding: 1px 5px;
    color: var(--accent-bright);
  }
  .doc-body :global(kbd) {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 11px;
    background: var(--panel-3);
    border: 1px solid var(--line-strong);
    border-bottom-width: 2px;
    border-radius: 4px;
    padding: 1px 6px;
    color: var(--text);
  }
  .doc-body :global(table) {
    width: 100%;
    border-collapse: collapse;
    margin: 4px 0 16px;
    font-size: 13px;
  }
  .doc-body :global(th), .doc-body :global(td) {
    text-align: left;
    padding: 7px 10px;
    border-bottom: 1px solid var(--line);
    vertical-align: top;
  }
  .doc-body :global(th) {
    color: var(--text-faint);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
  }
  .doc-body :global(td:first-child) { color: var(--text); white-space: nowrap; }
  .doc-body :global(.callout) {
    display: block;
    margin: 4px 0 16px;
    padding: 10px 14px;
    border-radius: var(--radius);
    border: 1px solid var(--line);
    border-left-width: 3px;
    font-size: 13px;
    line-height: 1.55;
    background: var(--panel-2);
  }
  .doc-body :global(.callout.tip) { border-left-color: var(--accent); }
  .doc-body :global(.callout.tip strong) { color: var(--accent-bright); }
  .doc-body :global(.callout.note) { border-left-color: var(--text-faint); }
  .doc-body :global(.callout.warn) { border-left-color: var(--warn); }
  .doc-body :global(.callout.warn strong) { color: var(--warn); }

  @media (max-width: 860px) {
    .help { grid-template-columns: 1fr; }
    .sidebar { display: none; }
  }
</style>
