from PIL import Image
import colorgram

# Function to convert the image to grayscale and save it
def convert_to_greyscale(image_number):
    # Dynamic image filename for .tiff images
    tiff_image_path = f'Sample_{image_number}.tiff'
    grey_image_path = f'greyscale_image_{image_number}.png'  # File to save the greyscale image

    # Open the original .tiff image and convert it to grayscale
    img = Image.open(tiff_image_path)
    grey_img = img.convert("L")  # Convert image to grayscale ('L' mode stands for 8-bit pixels, black and white)

    # Save the grayscale image
    grey_img.save(grey_image_path)

    return grey_img, grey_image_path

# Function to process each greyscale image and extract colors
def process_image(image_number):
    # Convert the original image to grayscale and get the saved grayscale image path
    grey_img, grey_image_path = convert_to_greyscale(image_number)

    # Use the grayscale image in colorgram for color extraction
    colors = colorgram.extract(grey_image_path, 30)  # Extract 30 colors

    # Loop through each color and get the RGB values (even though it's grayscale, colorgram will still extract shades)
    rgb_colors = []
    for color in colors:
        r = color.rgb.r
        g = color.rgb.g
        b = color.rgb.b
        rgb_colors.append((r, g, b))

    return rgb_colors

# File to store the extracted RGB colors
output_file = "extracted_colors.txt"

# Open the file in append mode (creates the file if it doesn't exist)
with open(output_file, "a") as file:
    for i in range(1, 12):  # Loop from Sample_1.tiff to Sample_11.tiff
        rgb_colors = process_image(i)
        
        # Write the image number and corresponding RGB colors to the file
        file.write(f"Colors from Sample_{i} (Greyscale).tiff:\n")
        for color in rgb_colors:
            # Append "rgb" in front of the RGB tuple
            file.write(f"rgb{color}\n")
        file.write("\n")  # Add a new line after each image's colors

print("Greyscale images saved, RGB colors extracted and saved to", output_file)
