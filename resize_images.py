#!/usr/bin/env python3
"""Resize images, recolor backgrounds, and overlay controllable crack patterns."""

from __future__ import annotations

import argparse
import hashlib
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
CROP_MARGIN = 25
FILL_COLOR = (255, 228, 225)
BLACK_THRESHOLD = 40
WHITE_TOLERANCE = 12
BACKGROUND_TOLERANCE = 8


@dataclass
class CrackSettings:
    """Parameters controlling the crack mesh density."""

    min_area: int = 1500
    max_area: int = 2000
    cell_size: Optional[int] = None
    variation: Optional[float] = None
    jitter: Optional[int] = None

    def resolved(self) -> "CrackSettings":
        target_area = max(1.0, (self.min_area + self.max_area) / 2.0)

        cell = self.cell_size
        if cell is None or cell <= 0:
            cell = max(8, int(round(target_area ** 0.5)))

        variation = self.variation
        if variation is None or variation <= 0:
            variation = min(0.25, max(0.02, (self.max_area - self.min_area) / (2 * target_area)))

        jitter = self.jitter
        if jitter is None or jitter < 0:
            jitter = max(1, int(round(cell * 0.08)))

        return CrackSettings(
            min_area=self.min_area,
            max_area=self.max_area,
            cell_size=cell,
            variation=variation,
            jitter=jitter,
        )

    def suffix(self) -> str:
        resolved = self.resolved()
        variation_str = f"{resolved.variation:.3f}".rstrip("0").rstrip(".")
        return (
            f"-minA{resolved.min_area}-maxA{resolved.max_area}-"
            f"c{resolved.cell_size}-V{variation_str}-J{resolved.jitter}"
        )


DEFAULT_SETTINGS = CrackSettings()


def get_lanczos_filter() -> int:
    try:
        return Image.Resampling.LANCZOS  # Pillow >= 9.1.0
    except AttributeError:
        return Image.LANCZOS  # Pillow < 9.1.0


def ensure_positive(value: int, name: str) -> int:
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be a positive integer, got {value}.")
    return value


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def is_black(color: tuple[int, int, int]) -> bool:
    return all(channel <= BLACK_THRESHOLD for channel in color)


def is_white(color: tuple[int, int, int]) -> bool:
    return all(channel >= 255 - WHITE_TOLERANCE for channel in color)


def colors_match(color: tuple[int, int, int], target: tuple[int, int, int]) -> bool:
    return all(abs(channel - t) <= BACKGROUND_TOLERANCE for channel, t in zip(color, target))


def crop_with_margin(image: Image.Image, margin: int) -> Image.Image:
    new_width = image.width - 2 * margin
    new_height = image.height - 2 * margin
    if new_width <= 0 or new_height <= 0:
        raise ValueError(f"Crop margin {margin} is too large for image size {image.width}x{image.height}.")
    return image.crop((margin, margin, image.width - margin, image.height - margin))


def recolor_background(image: Image.Image, fill_color: tuple[int, int, int]) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")

    pixels = image.load()
    width, height = image.size

    if is_black(pixels[0, 0]):
        raise ValueError("Top-left pixel is considered black; cannot flood fill background.")

    visited = [[False] * width for _ in range(height)]
    queue: deque[tuple[int, int]] = deque([(0, 0)])

    while queue:
        x, y = queue.pop()
        if visited[y][x]:
            continue
        visited[y][x] = True

        current = pixels[x, y]
        if is_black(current):
            continue

        if current != fill_color:
            pixels[x, y] = fill_color

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx]:
                queue.append((nx, ny))

    return image


def build_positions(
    limit: int, rng: random.Random, cell_size: int, variation: float
) -> list[float]:
    min_step = max(4.0, cell_size * (1 - variation))
    max_step = max(min_step + 1.0, cell_size * (1 + variation))

    positions = [0.0]
    while positions[-1] < limit:
        remaining = limit - positions[-1]
        if remaining <= min_step:
            positions.append(float(limit))
            break

        step = rng.uniform(min_step, max_step)
        if step > remaining:
            step = remaining
        positions.append(positions[-1] + step)

    if positions[-1] > limit:
        positions[-1] = float(limit)

    return positions


def generate_crack_mask(
    size: tuple[int, int],
    *,
    settings: CrackSettings,
    seed: str | None = None,
) -> Image.Image:
    width, height = size
    resolved = settings.resolved()

    seed_bytes = (seed or f"{width}x{height}").encode("utf-8")
    rng_seed = int.from_bytes(hashlib.blake2s(seed_bytes, digest_size=8).digest(), "big")
    rng = random.Random(rng_seed)

    x_positions = build_positions(width - 1, rng, resolved.cell_size, resolved.variation)
    y_positions = build_positions(height - 1, rng, resolved.cell_size, resolved.variation)

    points: list[list[tuple[int, int]]] = []
    for y in y_positions:
        row_points: list[tuple[int, int]] = []
        for x in x_positions:
            jittered_x = clamp(
                int(round(x + rng.uniform(-resolved.jitter, resolved.jitter))), 0, width - 1
            )
            jittered_y = clamp(
                int(round(y + rng.uniform(-resolved.jitter, resolved.jitter))), 0, height - 1
            )
            row_points.append((jittered_x, jittered_y))
        points.append(row_points)

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for row_points in points:
        for start, end in zip(row_points, row_points[1:]):
            draw.line([start, end], fill=255, width=1)

    for col in range(len(points[0])):
        column_points = [points[row][col] for row in range(len(points))]
        for start, end in zip(column_points, column_points[1:]):
            draw.line([start, end], fill=255, width=1)

    return mask


def mark_eligible_white_pixels(
    image: Image.Image, fill_color: tuple[int, int, int], min_area: int
) -> list[list[bool]]:
    width, height = image.size
    pixels = image.load()
    visited = [[False] * width for _ in range(height)]
    eligible = [[False] * width for _ in range(height)]

    for y in range(height):
        for x in range(width):
            if visited[y][x]:
                continue

            color = pixels[x, y]
            if not (is_white(color) and not colors_match(color, fill_color)):
                visited[y][x] = True
                continue

            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited[y][x] = True
            component: list[tuple[int, int]] = []

            while queue:
                cx, cy = queue.popleft()
                current = pixels[cx, cy]
                if not (is_white(current) and not colors_match(current, fill_color)):
                    continue

                component.append((cx, cy))

                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx]:
                        visited[ny][nx] = True
                        queue.append((nx, ny))

            if len(component) >= min_area:
                for cx, cy in component:
                    eligible[cy][cx] = True

    return eligible


def collect_component_areas(
    image: Image.Image,
    fill_color: tuple[int, int, int],
    min_area: int,
) -> list[int]:
    width, height = image.size
    pixels = image.load()
    visited = [[False] * width for _ in range(height)]
    areas: list[int] = []

    for y in range(height):
        for x in range(width):
            if visited[y][x]:
                continue
            visited[y][x] = True

            color = pixels[x, y]
            if not (is_white(color) and not colors_match(color, fill_color)):
                continue

            queue: deque[tuple[int, int]] = deque([(x, y)])
            area = 0

            while queue:
                cx, cy = queue.popleft()
                current = pixels[cx, cy]
                if not (is_white(current) and not colors_match(current, fill_color)):
                    continue

                area += 1

                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx]:
                        visited[ny][nx] = True
                        queue.append((nx, ny))

            if area >= min_area:
                areas.append(area)

    return areas


def apply_cracks(
    image: Image.Image,
    *,
    fill_color: tuple[int, int, int],
    seed: str | None = None,
    settings: CrackSettings,
) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")

    resolved = settings.resolved()
    eligible_map = mark_eligible_white_pixels(image, fill_color, resolved.min_area)
    if not any(any(row) for row in eligible_map):
        return image

    original = image.copy()
    best_image = image
    best_stats: Optional[tuple[int, int]] = None

    for attempt in range(6):
        scale = max(0.6, 1 - 0.08 * attempt)
        attempt_cell = max(5, int(round(resolved.cell_size * scale)))
        attempt_variation = max(0.01, min(0.35, resolved.variation + attempt * 0.015))
        attempt_jitter = max(1, resolved.jitter + attempt // 2)

        pattern = generate_crack_mask(
            original.size,
            settings=CrackSettings(
                min_area=resolved.min_area,
                max_area=resolved.max_area,
                cell_size=attempt_cell,
                variation=attempt_variation,
                jitter=attempt_jitter,
            ),
            seed=f"{seed or 'default'}:{attempt}",
        )

        trial = original.copy()
        mask_pixels = pattern.load()
        trial_pixels = trial.load()
        width, height = trial.size

        for y in range(height):
            eligible_row = eligible_map[y]
            for x in range(width):
                if mask_pixels[x, y] and eligible_row[x]:
                    trial_pixels[x, y] = (0, 0, 0)

        areas = collect_component_areas(trial, fill_color, resolved.min_area)
        if areas:
            min_area_val = min(areas)
            max_area_val = max(areas)
        else:
            min_area_val = max_area_val = 0

        if (
            areas
            and min_area_val >= resolved.min_area
            and max_area_val <= resolved.max_area
        ):
            return trial

        if best_stats is None or max_area_val < best_stats[1] or (
            max_area_val == best_stats[1] and min_area_val > best_stats[0]
        ):
            best_image = trial
            best_stats = (min_area_val, max_area_val)

    return best_image


def unique_destination(path: Path, *, suffix: str = "") -> Path:
    if not path.exists():
        return path

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    stem = path.stem
    suffix_part = f"{suffix}" if suffix else ""
    candidate = path.with_name(f"{stem}_{timestamp}{suffix_part}{path.suffix}")

    counter = 1
    while candidate.exists():
        candidate = path.with_name(
            f"{stem}_{timestamp}_{counter}{suffix_part}{path.suffix}"
        )
        counter += 1

    return candidate


def generate_destination_name(source: Path, destination_dir: Path, settings: CrackSettings) -> Path:
    suffix = settings.suffix()
    base = destination_dir / f"{source.stem}{suffix}{source.suffix}"
    return unique_destination(base)


def resize_images(
    source_dir: Path,
    destination_dir: Path,
    size: tuple[int, int],
    *,
    settings: CrackSettings,
) -> int:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder '{source_dir}' does not exist.")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path '{source_dir}' is not a directory.")

    destination_dir.mkdir(parents=True, exist_ok=True)

    resample_filter = get_lanczos_filter()
    processed = 0
    skipped: list[str] = []

    resolved = settings.resolved()

    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped.append(path.name)
            continue

        with Image.open(path) as img:
            processed_img = img.resize(size, resample=resample_filter)
            processed_img = crop_with_margin(processed_img, CROP_MARGIN)
            processed_img = recolor_background(processed_img, FILL_COLOR)
            processed_img = apply_cracks(
                processed_img,
                fill_color=FILL_COLOR,
                seed=path.name,
                settings=resolved,
            )

            if path.suffix.lower() in {".jpg", ".jpeg"} and processed_img.mode != "RGB":
                processed_img = processed_img.convert("RGB")

            save_kwargs = {}
            if path.suffix.lower() in {".jpg", ".jpeg"}:
                save_kwargs["quality"] = 95
            elif path.suffix.lower() == ".png":
                save_kwargs["optimize"] = True

            destination_path = generate_destination_name(path, destination_dir, resolved)
            processed_img.save(destination_path, **save_kwargs)
            processed += 1

    if skipped:
        skipped_names = ", ".join(skipped)
        print(f"Skipped non-image files: {skipped_names}")

    print(f"Processed {processed} image(s) into '{destination_dir}'.")
    return processed


def launch_gui(initial: CrackSettings) -> CrackSettings:
    import tkinter as tk
    from tkinter import messagebox

    resolved = initial.resolved()

    root = tk.Tk()
    root.title("Crack Mesh Settings")

    entries: dict[str, tk.Entry] = {}

    def add_row(label: str, value: str, row: int):
        tk.Label(root, text=label, anchor="w").grid(row=row, column=0, sticky="we", padx=6, pady=4)
        entry = tk.Entry(root)
        entry.insert(0, value)
        entry.grid(row=row, column=1, padx=6, pady=4)
        entries[label] = entry

    add_row("Minimum area", str(resolved.min_area), 0)
    add_row("Maximum area", str(resolved.max_area), 1)
    add_row("Cell size", str(resolved.cell_size), 2)
    add_row("Variation", f"{resolved.variation:.3f}", 3)
    add_row("Jitter", str(resolved.jitter), 4)

    result: dict[str, CrackSettings] = {}

    def on_submit() -> None:
        try:
            min_area = int(entries["Minimum area"].get())
            max_area = int(entries["Maximum area"].get())
            if min_area <= 0 or max_area <= 0 or max_area < min_area:
                raise ValueError
            cell_size = int(entries["Cell size"].get())
            variation = float(entries["Variation"].get())
            jitter = int(entries["Jitter"].get())
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Please enter positive numbers (max must be >= min).",
            )
            return

        result["settings"] = CrackSettings(
            min_area=min_area,
            max_area=max_area,
            cell_size=cell_size,
            variation=variation,
            jitter=jitter,
        )
        root.destroy()

    def on_cancel() -> None:
        root.destroy()

    button_frame = tk.Frame(root)
    button_frame.grid(row=5, column=0, columnspan=2, pady=8)
    tk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=4)
    tk.Button(button_frame, text="Start", command=on_submit).pack(side=tk.RIGHT, padx=4)

    root.mainloop()

    if "settings" not in result:
        raise SystemExit("Processing cancelled from GUI.")
    return result["settings"].resolved()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize images from one folder to another and apply a cracked mesh."
    )
    parser.add_argument("--source", "-s", default="alap", help="Source folder containing images.")
    parser.add_argument(
        "--destination",
        "-d",
        default="kesz",
        help="Destination folder to write resized images.",
    )
    parser.add_argument("--width", type=lambda v: ensure_positive(int(v), "width"), default=1024)
    parser.add_argument("--height", type=lambda v: ensure_positive(int(v), "height"), default=1024)

    parser.add_argument("--min-area", type=int, help="Minimum cracked component area.")
    parser.add_argument("--max-area", type=int, help="Maximum cracked component area.")
    parser.add_argument("--cell-size", type=int, help="Base grid cell size in pixels.")
    parser.add_argument(
        "--variation",
        type=float,
        help="Relative variation (0-0.5) applied when spacing grid lines.",
    )
    parser.add_argument("--jitter", type=int, help="Maximum random jitter applied to line vertices.")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable the GUI and rely entirely on CLI arguments.",
    )
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> CrackSettings:
    settings = CrackSettings()
    if args.min_area is not None:
        settings.min_area = max(1, args.min_area)
    if args.max_area is not None:
        settings.max_area = max(settings.min_area, args.max_area)
    if args.cell_size is not None:
        settings.cell_size = max(1, args.cell_size)
    if args.variation is not None:
        settings.variation = max(0.0, args.variation)
    if args.jitter is not None:
        settings.jitter = max(0, args.jitter)
    return settings


def main() -> None:
    args = parse_args()
    settings = build_settings(args)

    if args.no_gui:
        settings = settings.resolved()
    else:
        settings = launch_gui(settings)

    size = (args.width, args.height)
    resize_images(
        Path(args.source),
        Path(args.destination),
        size,
        settings=settings,
    )


if __name__ == "__main__":
    main()
