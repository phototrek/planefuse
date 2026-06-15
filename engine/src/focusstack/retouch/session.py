"""Deterministic retouch session with checkpointed undo and redo."""

from __future__ import annotations

import numpy as np

from focusstack.retouch.brush import EMPTY_BBOX, Stroke, stroke_mask

CHECKPOINT_EVERY = 20

Bbox = tuple[int, int, int, int]


class RetouchSession:
    def __init__(self, base: np.ndarray, sources: dict[str, np.ndarray]):
        self.base = np.ascontiguousarray(base, dtype=np.float32)
        self.sources = {
            source_id: np.ascontiguousarray(source, dtype=np.float32)
            for source_id, source in sources.items()
        }
        self.composite = self.base.copy()
        self.strokes: list[Stroke] = []
        self.redo_stack: list[Stroke] = []
        self.checkpoints: list[tuple[int, np.ndarray]] = [(0, self.base.copy())]

    def _blend(self, stroke: Stroke) -> Bbox:
        h, w = self.base.shape[:2]
        mask, bbox = stroke_mask(h, w, stroke)
        y0, x0, y1, x1 = bbox
        if bbox == EMPTY_BBOX:
            return bbox

        alpha = (mask * np.float32(stroke.opacity))[..., None]
        source = self.base if stroke.mode == "erase" else self.sources[stroke.source_id]
        region = self.composite[y0:y1, x0:x1]
        self.composite[y0:y1, x0:x1] = (
            region * (1.0 - alpha) + source[y0:y1, x0:x1] * alpha
        )
        return bbox

    def _commit(self, stroke: Stroke) -> Bbox:
        bbox = self._blend(stroke)
        if bbox == EMPTY_BBOX:
            return bbox
        self.strokes.append(stroke)
        if len(self.strokes) % CHECKPOINT_EVERY == 0:
            self.checkpoints.append((len(self.strokes), self.composite.copy()))
        return bbox

    def apply(self, stroke: Stroke) -> Bbox:
        bbox = self._commit(stroke)
        if bbox != EMPTY_BBOX:
            self.redo_stack.clear()
        return bbox

    def _rebuild_to(self, count: int) -> None:
        checkpoint_count, checkpoint = max(
            (item for item in self.checkpoints if item[0] <= count),
            key=lambda item: item[0],
        )
        self.composite = checkpoint.copy()
        for stroke in self.strokes[checkpoint_count:count]:
            self._blend(stroke)

    def undo(self) -> Bbox:
        if not self.strokes:
            return EMPTY_BBOX
        stroke = self.strokes.pop()
        self.redo_stack.append(stroke)
        self.checkpoints = [
            checkpoint for checkpoint in self.checkpoints if checkpoint[0] <= len(self.strokes)
        ]
        self._rebuild_to(len(self.strokes))
        h, w = self.base.shape[:2]
        return stroke_mask(h, w, stroke)[1]

    def redo(self) -> Bbox:
        if not self.redo_stack:
            return EMPTY_BBOX
        return self._commit(self.redo_stack.pop())
