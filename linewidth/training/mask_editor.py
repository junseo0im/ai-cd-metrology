"""Interactive editor for correcting U-Net binary masks.

Controls
--------
Left-click/drag: add mask, right-click/drag: erase mask
1: exact 1-pixel pencil, [ / ]: smaller/larger brush
z: undo, r: reset to U-Net prediction
s: save, n: save and next, p: save and previous, m: toggle overlay, q: save and quit
Mouse wheel: zoom around cursor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import MANUAL_REVIEW_ROOT


BASE = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = MANUAL_REVIEW_ROOT / "full_mask" / "review_manifest.json"


class MaskEditor:
    def __init__(self, manifest_path: Path, start_index: int = 0) -> None:
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.entries = self.manifest["entries"]
        if not self.entries:
            raise ValueError("review_manifest.json contains no images")
        self.index = min(max(0, start_index), len(self.entries) - 1)
        # Radius 0 changes exactly one native-mask pixel per click.
        self.brush_radius = 0
        self.mask_visible = True
        self.drawing = False
        self.draw_value = True
        self.undo_stack: list[np.ndarray] = []

        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        try:
            dataset_name = self.manifest_path.parent.name
            self.fig.canvas.manager.set_window_title(
                f"Mask Editor - 1px Pencil - {dataset_name}"
            )
        except AttributeError:
            pass
        self.image_artist = None
        self.mask_artist = None
        self.status_artist = None
        self.image = None
        self.initial_mask = None
        self.mask = None

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.load_current()

    @property
    def entry(self) -> dict:
        return self.entries[self.index]

    def load_current(self) -> None:
        self.image = np.asarray(Image.open(self.entry["native_image"]).convert("RGB"))
        self.initial_mask = np.asarray(Image.open(self.entry["initial_mask"]).convert("L")) > 127
        corrected_path = Path(self.entry["corrected_mask"])
        if corrected_path.exists():
            corrected = np.asarray(Image.open(corrected_path).convert("L")) > 127
            if corrected.shape != self.initial_mask.shape:
                raise ValueError(f"Corrected mask size mismatch: {corrected_path}")
            self.mask = corrected.copy()
        else:
            self.mask = self.initial_mask.copy()
        self.undo_stack.clear()
        self.redraw(reset_view=True)

    def overlay_rgba(self) -> np.ndarray:
        height, width = self.mask.shape
        overlay = np.zeros((height, width, 4), dtype=np.float32)
        overlay[..., 1] = 1.0
        overlay[..., 2] = 1.0
        overlay[..., 3] = self.mask.astype(np.float32) * (0.55 if self.mask_visible else 0.0)
        return overlay

    def redraw(self, reset_view: bool = False) -> None:
        # Thin horizontal bands are stretched vertically on screen so that
        # individual native pixels remain easy to click.  The mask array and
        # saved coordinates are not resized.
        display_aspect = "auto" if self.image.shape[0] <= 128 else "equal"
        if self.image_artist is None:
            self.image_artist = self.ax.imshow(self.image, aspect=display_aspect)
            self.mask_artist = self.ax.imshow(
                self.overlay_rgba(), interpolation="nearest", aspect=display_aspect
            )
            self.status_artist = self.ax.text(
                0.01,
                0.01,
                "",
                transform=self.ax.transAxes,
                color="white",
                fontsize=10,
                va="bottom",
                bbox=dict(facecolor="black", alpha=0.65, pad=4),
            )
        else:
            self.image_artist.set_data(self.image)
            self.mask_artist.set_data(self.overlay_rgba())
            self.ax.set_aspect(display_aspect)
        changed = int(np.count_nonzero(self.mask != self.initial_mask))
        title = f"[{self.index + 1}/{len(self.entries)}] {self.entry['id']} — cyan=mask"
        self.ax.set_title(title)
        brush_label = "pencil=1px" if self.brush_radius == 0 else f"brush radius={self.brush_radius}px"
        self.status_artist.set_text(
            f"{brush_label}  changed={changed}px | left add, right erase | "
            "1 exact pixel, [ ] brush, z undo, r reset, s save, n/p next/prev, m mask, q quit"
        )
        if reset_view:
            self.ax.set_xlim(-0.5, self.image.shape[1] - 0.5)
            self.ax.set_ylim(self.image.shape[0] - 0.5, -0.5)
        self.fig.canvas.draw_idle()

    def paint(self, x: float, y: float) -> None:
        cx, cy = int(round(x)), int(round(y))
        height, width = self.mask.shape
        if not (0 <= cx < width and 0 <= cy < height):
            return
        if self.brush_radius == 0:
            self.mask[cy, cx] = self.draw_value
            self.mask_artist.set_data(self.overlay_rgba())
            self.fig.canvas.draw_idle()
            return
        x0, x1 = max(0, cx - self.brush_radius), min(width, cx + self.brush_radius + 1)
        y0, y1 = max(0, cy - self.brush_radius), min(height, cy + self.brush_radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= self.brush_radius ** 2
        target = self.mask[y0:y1, x0:x1]
        target[disk] = self.draw_value
        self.mask_artist.set_data(self.overlay_rgba())
        self.fig.canvas.draw_idle()

    def on_press(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button not in (1, 3):
            return
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)
        self.drawing = True
        self.draw_value = event.button == 1
        self.paint(event.xdata, event.ydata)

    def on_motion(self, event) -> None:
        if not self.drawing or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.paint(event.xdata, event.ydata)

    def on_release(self, _event) -> None:
        self.drawing = False
        self.redraw()

    def save_current(self) -> None:
        corrected_path = Path(self.entry["corrected_mask"])
        preview_path = Path(self.entry["corrected_preview"])
        corrected_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.mask.astype(np.uint8) * 255, mode="L").save(corrected_path)
        preview = self.image.astype(np.float32).copy()
        preview[self.mask] = preview[self.mask] * 0.45 + np.array([0, 255, 255]) * 0.55
        Image.fromarray(np.clip(preview, 0, 255).astype(np.uint8), mode="RGB").save(preview_path)
        self.status_artist.set_text(f"SAVED: {corrected_path}")
        self.fig.canvas.draw_idle()

    def switch(self, step: int) -> None:
        self.save_current()
        self.index = (self.index + step) % len(self.entries)
        self.load_current()

    def on_key(self, event) -> None:
        if event.key == "1":
            self.brush_radius = 0
        elif event.key == "[":
            self.brush_radius = max(0, self.brush_radius - 1)
        elif event.key == "]":
            self.brush_radius = min(50, self.brush_radius + 1)
        elif event.key == "z" and self.undo_stack:
            self.mask = self.undo_stack.pop()
        elif event.key == "r":
            self.undo_stack.append(self.mask.copy())
            self.mask = self.initial_mask.copy()
        elif event.key == "s":
            self.save_current()
        elif event.key == "n":
            self.switch(1)
            return
        elif event.key == "p":
            self.switch(-1)
            return
        elif event.key == "m":
            self.mask_visible = not self.mask_visible
        elif event.key == "q":
            self.save_current()
            plt.close(self.fig)
            return
        self.redraw()

    def on_scroll(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        new_w = (x1 - x0) * scale
        new_h = (y1 - y0) * scale
        rx = (event.xdata - x0) / (x1 - x0)
        ry = (event.ydata - y0) / (y1 - y0)
        self.ax.set_xlim(event.xdata - new_w * rx, event.xdata + new_w * (1 - rx))
        self.ax.set_ylim(event.ydata - new_h * ry, event.ydata + new_h * (1 - ry))
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.tight_layout()
        plt.show()


def check_manifest(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []
    for entry in manifest.get("entries", []):
        image_path = Path(entry["native_image"])
        mask_path = Path(entry["initial_mask"])
        if not image_path.exists() or not mask_path.exists():
            problems.append(f"missing file: {entry['id']}")
            continue
        image_size = Image.open(image_path).size
        mask_size = Image.open(mask_path).size
        if image_size != mask_size:
            problems.append(f"size mismatch {entry['id']}: image={image_size}, mask={mask_size}")
    print("Images:", len(manifest.get("entries", [])))
    print("Status:", "PASS" if not problems else "FAIL")
    print("Problems:", problems)
    if problems:
        raise RuntimeError("Mask review manifest validation failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Edit predicted U-Net masks")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--index", type=int, default=1, help="1-based image index")
    parser.add_argument("--check", action="store_true", help="Validate files without opening GUI")
    args = parser.parse_args()
    if args.check:
        check_manifest(args.manifest)
        return
    MaskEditor(args.manifest, start_index=args.index - 1).show()


if __name__ == "__main__":
    main()

