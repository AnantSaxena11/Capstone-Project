from PIL import Image
import os

# Current directory
current_directory = os.getcwd()

# Function to check if a pixel is white
def is_white(pixel, tolerance=10):
    return all(255 - tolerance <= channel <= 255 for channel in pixel)

# Process images from 1 to 11
for i in range(1, 12):
    # Generate input and output paths
    input_image_path = os.path.join(current_directory, f'marked_canvas_gray_greyscale_image_{i}.png')
    output_image_path = os.path.join(current_directory, f'marked_boundary_image_{i}.png')
    output_text_path = os.path.join(current_directory, f'boundary_pixels_{i}.txt')
    
    # Load the image
    image = Image.open(input_image_path).convert('RGB')
    width, height = image.size

    # Create a copy of the image to mark boundaries
    new_image = image.copy()
    boundary_pixels = set()  # Using a set to avoid duplicate boundary pixels

    # Scan the image pixel by pixel to detect boundaries
    for y in range(height):
        for x in range(width):
            current_pixel = image.getpixel((x, y))

            # Check transitions to non-white pixels in the surrounding 4 neighbors (left, right, top, bottom)
            neighbors = []
            if x > 0:  # Check left pixel
                neighbors.append(image.getpixel((x - 1, y)))
            if x < width - 1:  # Check right pixel
                neighbors.append(image.getpixel((x + 1, y)))
            if y > 0:  # Check top pixel
                neighbors.append(image.getpixel((x, y - 1)))
            if y < height - 1:  # Check bottom pixel
                neighbors.append(image.getpixel((x, y + 1)))

            # Mark pixel if there is a transition from white to non-white or vice versa
            if any(is_white(neighbor) != is_white(current_pixel) for neighbor in neighbors):
                # Mark boundary in blue and record the coordinate
                new_image.putpixel((x, y), (0, 0, 255))  # Mark in blue
                boundary_pixels.add((x, y))

    # Save the modified image with marked boundaries
    new_image.save(output_image_path)

    # Write the boundary pixel coordinates to a text file
    with open(output_text_path, 'w') as f:
        for coord in sorted(boundary_pixels):  # Sort for consistency
            f.write(f"{coord}\n")

    print(f"Processing completed for {input_image_path}. Results saved to {output_image_path} and {output_text_path}")
