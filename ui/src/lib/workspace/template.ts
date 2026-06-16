// Client-side filename templating for export (SPEC §130).
// ponytail: 5 fixed tokens, plain string replace — no template-engine dep.
export interface TemplateCtx {
  stack_name: string;
  method: string;
  frames: number | undefined;
  date: string;   // YYYY-MM-DD
  seq: number;    // 1-based
}

const ILLEGAL = /[\\/:*?"<>|]/g;

export function applyTemplate(template: string, ctx: TemplateCtx): string {
  const values: Record<string, string> = {
    stack_name: ctx.stack_name,
    method: ctx.method,
    frames: ctx.frames === undefined ? '' : String(ctx.frames),
    date: ctx.date,
    seq: String(ctx.seq).padStart(3, '0')
  };
  const out = template.replace(/\{(\w+)\}/g, (m, key) =>
    key in values ? values[key] : m // unknown token stays literal
  );
  return out.replace(ILLEGAL, '_');
}

export function todayISO(): string {
  return new Date().toLocaleDateString('sv-SE'); // 'sv-SE' → YYYY-MM-DD, local tz
}
