#!/usr/bin/env python3
"""Resize images, crop borders, recolor background, and add uniform cracks."""

import argparse
import hashlib
import random
from collections import deque
from pathlib import Path

from PIL import Image, ImageDraw


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
CROP_MARGIN = 25
FILL_COLOR = (255, 228, 225)
BLACK_THRESHOLD = 40
WHITE_TOLERANCE = 12
BACKGROUND_TOLERANCE = 8
MIN_COMPONENT_AREA = 1500
GRID_CELL_SIZE = 41
GRID_VARIATION = 0.05
GRID_JITTER = 3


def get_lanczos_filter() -> int:
    """Return the appropriate LANCZOS resampling filter depending on Pillow version."""
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


def build_positions(limit: int, rng: random.Random) -> list[float]:
    positions = [0.0]
    min_step = GRID_CELL_SIZE * (1 - GRID_VARIATION)
    max_step = GRID_CELL_SIZE * (1 + GRID_VARIATION)

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


def generate_crack_mask(size: tuple[int, int], seed: str | None = None) -> Image.Image:
    width, height = size
    seed_bytes = (seed or f"{width}x{height}").encode("utf-8")
    rng_seed = int.from_bytes(hashlib.blake2s(seed_bytes, digest_size=8).digest(), "big")
    rng = random.Random(rng_seed)

    x_positions = build_positions(width - 1, rng)
    y_positions = build_positions(height - 1, rng)

    points: list[list[tuple[int, int]]] = []
    for y in y_positions:
        row_points: list[tuple[int, int]] = []
        for x in x_positions:
            jittered_x = clamp(int(round(x + rng.uniform(-GRID_JITTER, GRID_JITTER))), 0, width - 1)
            jittered_y = clamp(int(round(y + rng.uniform(-GRID_JITTER, GRID_JITTER))), 0, height - 1)
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


def apply_cracks(
    image: Image.Image, fill_color: tuple[int, int, int], seed: str | None = None
) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")

    eligible = mark_eligible_white_pixels(image, fill_color, MIN_COMPONENT_AREA)
    mask = generate_crack_mask(image.size, seed=seed)
    mask_pixels = mask.load()
    pixels = image.load()

    width, height = image.size
    for y in range(height):
        for x in range(width):
            if mask_pixels[x, y] and eligible[y][x]:
                pixels[x, y] = (0, 0, 0)

    return image


def resize_images(source_dir: Path, destination_dir: Path, size: tuple[int, int]) -> int:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder '{source_dir}' does not exist.")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path '{source_dir}' is not a directory.")

    destination_dir.mkdir(parents=True, exist_ok=True)

    resample_filter = get_lanczos_filter()
    processed = 0
    skipped: list[str] = []

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
            processed_img = apply_cracks(processed_img, fill_color=FILL_COLOR, seed=path.name)

            if path.suffix.lower() in {".jpg", ".jpeg"} and processed_img.mode != "RGB":
                processed_img = processed_img.convert("RGB")

            save_kwargs = {}
            if path.suffix.lower() in {".jpg", ".jpeg"}:
                save_kwargs["quality"] = 95
            elif path.suffix.lower() == ".png":
                save_kwargs["optimize"] = True

            processed_img.save(destination_dir / path.name, **save_kwargs)
            processed += 1

    if skipped:
        skipped_names = ", ".join(skipped)
        print(f"Skipped non-image files: {skipped_names}")

    print(f"Processed {processed} image(s) into '{destination_dir}'.")
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize images from one folder to another.")
    parser.add_argument("--source", "-s", default="alap", help="Source folder containing images.")
    parser.add_argument(
        "--destination",
        "-d",
        default="kesz",
        help="Destination folder to write resized images.",
    )
    parser.add_argument("--width", type=lambda v: ensure_positive(int(v), "width"), default=1024)
    parser.add_argument("--height", type=lambda v: ensure_positive(int(v), "height"), default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    size = (args.width, args.height)
    resize_images(Path(args.source), Path(args.destination), size)


if __name__ == "__main__":
    main()
