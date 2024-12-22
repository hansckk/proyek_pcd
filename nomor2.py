from PIL import Image
import numpy as np

input_image_path = 'ab.jpg'
image = Image.open(input_image_path)

image_rgb = image.convert("RGB")

orange_rgb = np.array([255, 128, 0]) / 255.0

def color_distance(c1, c2):
    return np.linalg.norm(c1 - c2)

width, height = image_rgb.size
quadrants = {
    1: image_rgb.crop((0, 0, width // 2, height // 2)),
    2: image_rgb.crop((width // 2, 0, width, height // 2)),
    3: image_rgb.crop((0, height // 2, width // 2, height)),
    4: image_rgb.crop((width // 2, height // 2, width, height)),
}

closest_quadrant = None
min_distance = float("inf")

for position, quadrant in quadrants.items():

    quadrant_array = np.array(quadrant) / 255.0

    mask = np.any(quadrant_array < 0.9, axis=-1)

    avg_color = quadrant_array[mask].mean(axis=0)

    distance = color_distance(avg_color, orange_rgb)

    if distance < min_distance:
        min_distance = distance
        closest_quadrant = position


print(f"Bentuk oranye paling dominan berada di posisi: {closest_quadrant}")
