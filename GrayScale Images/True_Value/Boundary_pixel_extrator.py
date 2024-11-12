from PIL import Image

# Function to check if a color is a shade of red (more lenient thresholds)
def is_red_shade(r, g, b, red_threshold=80, max_green_blue=120):
    """
    Determines if a color is a shade of red with a more lenient threshold.
    - red_threshold: Minimum value for the red component to qualify as red.
    - max_green_blue: Maximum values for green and blue components to include reddish shades.
    """
    return r > red_threshold and g < max_green_blue and b < max_green_blue

# Function to process each image and extract pixel locations of red shades
def process_image(image_number):
    # Dynamic image filename for .png images
    png_image_path = f'IMG{image_number}.png'
    
    # Open the .png image in RGB mode
    img = Image.open(png_image_path)
    img = img.convert("RGB")
    
    # Initialize a list to hold red pixel positions
    red_pixel_positions = []
    
    # Loop through each pixel in the image
    width, height = img.size
    for x in range(width):
        for y in range(height):
            r, g, b = img.getpixel((x, y))
            if is_red_shade(r, g, b):
                red_pixel_positions.append((x, y))
    
    return red_pixel_positions

# Process images and write pixel positions to separate text files
for i in range(1, 12):  # Loop from IMG1.png to IMG11.png
    red_pixel_positions = process_image(i)
    
    # Write the image number and corresponding red pixel locations to a file
    output_file = f"red_pixel_positions_IMG{i}.txt"
    with open(output_file, "w") as file:
        file.write(f"Red pixel positions from IMG{i}.png:\n")
        for position in red_pixel_positions:
            file.write(f"{position}\n")  # Write each (x, y) position

    print(f"Red pixel positions extracted and saved to {output_file}")
