from collections import deque
from PIL import Image

path = 'kesz/Friendly Pumpkin Coloring Sheet - Fall & Halloween Fun.png'
img = Image.open(path)
width, height = img.size
pixels = img.load()

WHITE_THRESHOLD = 248

def is_white(px):
    r, g, b = px
    return r >= WHITE_THRESHOLD and g >= WHITE_THRESHOLD and b >= WHITE_THRESHOLD

visited = [[False] * width for _ in range(height)]
components = []

for y in range(height):
    for x in range(width):
        if visited[y][x]:
            continue
        if not is_white(pixels[x, y]):
            visited[y][x] = True
            continue
        queue = deque([(x, y)])
        visited[y][x] = True
        area = 0
        min_x = max_x = x
        min_y = max_y = y
        while queue:
            cx, cy = queue.popleft()
            if not is_white(pixels[cx, cy]):
                continue
            area += 1
            if cx < min_x:
                min_x = cx
            if cx > max_x:
                max_x = cx
            if cy < min_y:
                min_y = cy
            if cy > max_y:
                max_y = cy
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx]:
                    visited[ny][nx] = True
                    if is_white(pixels[nx, ny]):
                        queue.append((nx, ny))
        components.append({
            'area': area,
            'bbox': (min_x, min_y, max_x, max_y),
            'center': ((min_x + max_x) / 2, (min_y + max_y) / 2),
        })

# Filter for components near top-left hillside area
filtered = [c for c in components if 0 <= c['center'][0] <= 400 and 300 <= c['center'][1] <= 500]
filtered.sort(key=lambda c: c['center'][1])

print(f'Filtered components ({len(filtered)} found) in x<=400, 300<=y<=500:')
for comp in filtered:
    print(comp)
