import os
import math
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import colorgram


# Ensure output directory exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Step 1: Convert Color Image to Greyscale
def convert_to_greyscale(image_number, output_dir):
    sample_images_folder = "Sample Images"
    tiff_image_path = os.path.join(sample_images_folder, f'Sample_{image_number}.tiff')
    grey_image_path = os.path.join(output_dir, 'greyscale_image.png')

    # Open and convert the image
    img = Image.open(tiff_image_path)
    grey_img = img.convert("L")
    grey_img.save(grey_image_path)

    return grey_image_path


# Step 2: Extract Gray Color Markings
LOW_GRAY_POINT = (31, 31, 31)
HIGH_GRAY_POINT = (83, 83, 83)


def is_gray_color_in_range(color, low, high):
    return low[0] <= color[0] <= high[0] and low[1] <= color[1] <= high[1] and low[2] <= color[2] <= high[2]


def extract_gray_markings(image_number, output_dir):
    image_name = os.path.join(output_dir, 'greyscale_image.png')
    img = Image.open(image_name).convert("RGB")
    width, height = img.size
    canvas = Image.new("RGB", (width, height), "white")

    for x in range(width):
        for y in range(height):
            color = img.getpixel((x, y))
            if is_gray_color_in_range(color, LOW_GRAY_POINT, HIGH_GRAY_POINT):
                canvas.putpixel((x, y), color)

    marked_image_path = os.path.join(output_dir, 'marked_canvas_gray.png')
    canvas.save(marked_image_path)
    return marked_image_path


# Step 3: Mark Boundaries in Blue
def is_white(pixel, tolerance=10):
    return all(255 - tolerance <= channel <= 255 for channel in pixel)


def mark_boundaries(image_number, output_dir):
    input_image_path = os.path.join(output_dir, 'marked_canvas_gray.png')
    output_image_path = os.path.join(output_dir, 'marked_boundary_image.png')
    output_text_path = os.path.join(output_dir, 'boundary_pixels.txt')

    image = Image.open(input_image_path).convert('RGB')
    width, height = image.size
    new_image = image.copy()
    boundary_pixels = set()

    for y in range(height):
        for x in range(width):
            current_pixel = image.getpixel((x, y))
            neighbors = []

            if x > 0:
                neighbors.append(image.getpixel((x - 1, y)))
            if x < width - 1:
                neighbors.append(image.getpixel((x + 1, y)))
            if y > 0:
                neighbors.append(image.getpixel((x, y - 1)))
            if y < height - 1:
                neighbors.append(image.getpixel((x, y + 1)))

            if any(is_white(neighbor) != is_white(current_pixel) for neighbor in neighbors):
                new_image.putpixel((x, y), (0, 0, 255))
                boundary_pixels.add((x, y))

    new_image.save(output_image_path)
    with open(output_text_path, 'w') as f:
        for coord in sorted(boundary_pixels):
            f.write(f"{coord}\n")

    return output_text_path


# Step 4: Extract Red Pixel Positions
def is_red_shade(r, g, b, red_threshold=80, max_green_blue=120):
    return r > red_threshold and g < max_green_blue and b < max_green_blue


def extract_red_pixels(image_number, output_dir):
    images_folder = "Images"
    image_path = os.path.join(images_folder, f'IMG{image_number}.png')

    img = Image.open(image_path).convert("RGB")
    red_pixel_positions = []

    width, height = img.size
    for x in range(width):
        for y in range(height):
            r, g, b = img.getpixel((x, y))
            if is_red_shade(r, g, b):
                red_pixel_positions.append((x, y))

    output_file = os.path.join(output_dir, "red_pixel_positions.txt")
    with open(output_file, "w") as file:
        for position in red_pixel_positions:
            file.write(f"{position}\n")

    return output_file


# Step 5: Calculate Error Percentage
def read_and_sort_coordinates(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(int, line.strip("()\n").split(","))
            coordinates.append((x, y))
    coordinates.sort()
    return coordinates


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_error_percentage(red_pixels_file, boundary_pixels_file, image_width, image_height):
    red_pixels = read_and_sort_coordinates(red_pixels_file)
    boundary_pixels = read_and_sort_coordinates(boundary_pixels_file)

    max_distance = math.sqrt(image_width ** 2 + image_height ** 2)
    total_distance = 0
    num_pairs = min(len(red_pixels), len(boundary_pixels))

    for i in range(num_pairs):
        total_distance += euclidean_distance(red_pixels[i], boundary_pixels[i])

    mean_distance = total_distance / num_pairs if num_pairs > 0 else 0
    return (mean_distance / max_distance) * 100


# Step 6: Compute and Plot Error Analysis
def process_images(num_images, image_width, image_height):
    results = {}

    for i in range(1, num_images + 1):
        output_dir = f'Image_{i}'
        ensure_directory_exists(output_dir)

        error_percentage = calculate_error_percentage(
            os.path.join(output_dir, "red_pixel_positions.txt"),
            os.path.join(output_dir, "boundary_pixels.txt"),
            image_width,
            image_height
        )
        results[f"IMG{i}"] = error_percentage
        print(f"Error Percentage for IMG{i}: {error_percentage:.2f}%")

    return results


def plot_error_analysis(error_percentages):
    images = list(error_percentages.keys())
    error_values = list(error_percentages.values())

    plt.figure(figsize=(10, 6))
    plt.plot(images, error_values, marker='o', color='skyblue', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('Image')
    plt.ylabel('Error Percentage (%)')
    plt.title('Error Percentage for Each Image')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Running the Full Process Synchronously
def main():
    num_images = 11
    image_width = 64
    image_height = 64

    for i in range(1, num_images + 1):
        output_dir = f'Image_{i}'
        ensure_directory_exists(output_dir)

        print(f"\nProcessing Image {i}...\n")
        convert_to_greyscale(i, output_dir)
        extract_gray_markings(i, output_dir)
        mark_boundaries(i, output_dir)
        extract_red_pixels(i, output_dir)

    error_percentages = process_images(num_images, image_width, image_height)
    plot_error_analysis(error_percentages)


if __name__ == "__main__":
    main()
