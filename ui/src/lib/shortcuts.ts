export function isEditableTarget(target: EventTarget | null): boolean {
  return target instanceof HTMLInputElement
    || target instanceof HTMLSelectElement
    || target instanceof HTMLTextAreaElement
    || (target instanceof HTMLElement && target.isContentEditable);
}

export function primaryShortcut(event: KeyboardEvent, key: string): boolean {
  return (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === key.toLowerCase();
}

export function plainShortcut(event: KeyboardEvent, key: string): boolean {
  return !event.metaKey && !event.ctrlKey && !event.altKey && event.key.toLowerCase() === key.toLowerCase();
}
