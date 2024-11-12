import os
from PIL import Image

# Define the low and high points for the "gray" color range
LOW_GRAY_POINT = (31, 31, 31)  # Low range for grayscale
HIGH_GRAY_POINT = (83, 83, 83)  # High range for grayscale

# Function to check if a color is within the specified range
def is_gray_color_in_range(color, low, high):
    r, g, b = color
    return low[0] <= r <= high[0] and low[1] <= g <= high[1] and low[2] <= b <= high[2]

# Function to extract gray colors and mark their pixel locations
def extract_gray_colors_and_mark():
    for image_number in range(1, 13):  # Loop through greyscale_image_1.png to greyscale_image_12.png
        # Define the path for the grayscale image
        image_name = f'greyscale_image_{image_number}.png'

        # Open the image
        try:
            img = Image.open(image_name)
        except FileNotFoundError:
            print(f"File not found: {image_name}. Skipping...")
            continue  # Skip this iteration if the file is not found

        img = img.convert("RGB")  # Ensure the image is in RGB mode

        # Get image dimensions
        width, height = img.size

        # Create a white canvas of the same size
        canvas = Image.new("RGB", (width, height), "white")

        # Open a text file to log pixel colors and locations
        output_text_path = f"pixel_colors_{os.path.splitext(image_name)[0]}.txt"
        with open(output_text_path, "w") as file:
            # Iterate through each pixel in the image
            for x in range(width):
                for y in range(height):
                    # Get the RGB value of the current pixel
                    color = img.getpixel((x, y))

                    # Write the pixel location and RGB value to the text file
                    file.write(f"Pixel ({x}, {y}): rgb({color[0]}, {color[1]}, {color[2]})\n")

                    # Check if the color is within the "gray" range
                    if is_gray_color_in_range(color, LOW_GRAY_POINT, HIGH_GRAY_POINT):
                        # Mark the pixel on the canvas with the same color as the original pixel
                        canvas.putpixel((x, y), color)  # Using the original color for gray pixels

        # Save the marked canvas as a new image
        output_canvas_path = f"marked_canvas_gray_{os.path.splitext(image_name)[0]}.png"
        canvas.save(output_canvas_path)
        print(f"Marked canvas saved as: {output_canvas_path}")
        print(f"Pixel colors and locations saved to: {output_text_path}")

# Run the function to extract gray colors and mark the pixels
extract_gray_colors_and_mark()
print("Gray color extraction and marking completed.")
