"""Native OS file/folder picker, run as a subprocess so Tk owns a clean main
thread and never touches the server's event loop.

Invoked as `python -m focusstack_server.picker <directory|files>`; prints a JSON
list of absolute paths to stdout (empty list if cancelled).

ponytail: subprocess + stdlib tkinter — no GUI framework, no threading dance.
Localhost desktop only (the dialog opens on the host running the server).
"""

from __future__ import annotations

import json
import sys


def pick(mode: str) -> list[str]:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if mode == "files":
            chosen = filedialog.askopenfilenames(
                title="Select image frames",
                filetypes=[("Images", "*.tif *.tiff *.jpg *.jpeg *.png"), ("All files", "*.*")],
            )
            paths = list(chosen)
        else:
            d = filedialog.askdirectory(title="Select a folder of frames")
            paths = [d] if d else []
    finally:
        root.destroy()
    return [str(p) for p in paths if p]


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "directory"
    print(json.dumps(pick(mode)))


if __name__ == "__main__":
    main()
