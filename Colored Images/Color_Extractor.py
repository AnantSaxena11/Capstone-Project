from PIL import Image
import colorgram

# Function to process each image and extract colors
def process_image(image_number):
    # Dynamic image filename for .tiff images
    tiff_image_path = f'Sample_{image_number}.tiff'
    
    # Open and convert the .tiff image to RGB mode
    img = Image.open(tiff_image_path)
    img = img.convert("RGB")
    
    # Instead of saving the image, pass the image object directly to colorgram
    colors = colorgram.extract(img, 30)  # Extract 30 colors
    
    # Loop through each color and get the RGB values
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
        file.write(f"Colors from Sample_{i}.tiff:\n")
        for color in rgb_colors:
            # Append "rgb" in front of the RGB tuple
            file.write(f"rgb{color}\n")
        file.write("\n")  # Add a new line after each image's colors

print("RGB colors extracted and saved to", output_file)
