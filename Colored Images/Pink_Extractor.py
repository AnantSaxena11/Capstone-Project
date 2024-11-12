import os
from PIL import Image

# Define the low and high points for the "pink" color range
LOW_POINT = (180, 100, 150)  # Adjusted Low range for RGB to capture lighter pinks
HIGH_POINT = (255, 200, 255)  # Adjusted High range for RGB to capture darker pinks

# Function to check if a color is within the specified range
def is_color_in_range(color, low, high):
    r, g, b = color
    return low[0] <= r <= high[0] and low[1] <= g <= high[1] and low[2] <= b <= high[2]

# Function to extract pink colors and mark their pixel locations on a white canvas
def extract_pink_colors_and_mark():
    for image_number in range(1, 12):  # Loop through Sample_1.tiff to Sample_11.tiff
        # Define the path for the TIFF image
        tiff_image_path = f'Sample_{image_number}.tiff'

        # Open the image
        try:
            img = Image.open(tiff_image_path)
        except FileNotFoundError:
            print(f"File not found: {tiff_image_path}. Skipping...")
            continue  # Skip this iteration if the file is not found
        
        img = img.convert("RGB")  # Ensure the image is in RGB mode
        
        # Get image dimensions
        width, height = img.size
        
        # Create a white canvas of the same size
        canvas = Image.new("RGB", (width, height), "white")
        
        # Open a text file to log pixel colors and locations
        with open(f"pixel_colors_Sample_{image_number}.txt", "w") as file:
            # Iterate through each pixel in the image
            for x in range(width):
                for y in range(height):
                    # Get the RGB value of the current pixel
                    color = img.getpixel((x, y))

                    # Check if the color is within the "pink" range
                    if is_color_in_range(color, LOW_POINT, HIGH_POINT):
                        # Mark the pixel on the canvas with the same color as the original pixel
                        canvas.putpixel((x, y), color)
                        
                        # Write the pixel location and color to the text file
                        file.write(f"Pixel ({x}, {y}): rgb({color[0]}, {color[1]}, {color[2]})\n")
        
        # Save the marked canvas as a new image
        output_canvas_path = f"marked_canvas_Sample_{image_number}.png"
        canvas.save(output_canvas_path)
        print(f"Marked canvas saved as: {output_canvas_path}")
        print(f"Pixel colors and locations saved to: pixel_colors_Sample_{image_number}.txt")

# Run the function to extract pink colors and mark the pixels
extract_pink_colors_and_mark()
print("All image processing completed.")
