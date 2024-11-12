import os
import math
from pathlib import Path
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Function to read and sort coordinates from a text file
def read_and_sort_coordinates(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            # Convert each line into a tuple of integers representing coordinates (x, y)
            x, y = map(int, line.strip("()\n").split(","))
            coordinates.append((x, y))
    
    # Sort the coordinates based on (x, y) for consistent order
    coordinates.sort()
    return coordinates

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to calculate error percentage
def calculate_error_percentage(red_pixels, boundary_pixels, image_width, image_height):
    # Calculate the diagonal (maximum possible distance) for normalization
    max_distance = math.sqrt(image_width ** 2 + image_height ** 2)
    total_distance = 0

    # Determine the number of pairs to compare based on the shorter list
    num_pairs = min(len(red_pixels), len(boundary_pixels))

    # Calculate the total Euclidean distance for each corresponding pair
    for i in range(num_pairs):
        total_distance += euclidean_distance(red_pixels[i], boundary_pixels[i])

    # Calculate mean absolute error if there are pairs to compare
    mean_distance = total_distance / num_pairs if num_pairs > 0 else 0
    error_percentage = (mean_distance / max_distance) * 100

    return error_percentage

# Main function to process all images and calculate error percentages
def process_images(num_images, image_width, image_height):
    results = {}
    # Use relative paths for current directory structure
    current_directory = Path(".")  # Refers to the current directory
    
    for i in range(1, num_images + 1):
        # Construct file paths for red and boundary pixel files in the current directory
        red_pixels_file = current_directory / f"red_pixel_positions_IMG{i}.txt"
        boundary_pixels_file = current_directory / f"boundary_pixels_{i}.txt"
        
        # Read and sort coordinates from the files
        red_pixels = read_and_sort_coordinates(red_pixels_file)
        boundary_pixels = read_and_sort_coordinates(boundary_pixels_file)
        
        # Calculate the error percentage
        error_percentage = calculate_error_percentage(red_pixels, boundary_pixels, image_width, image_height)
        results[f"IMG{i}"] = error_percentage
        print(f"Error Percentage for IMG{i}: {error_percentage:.2f}%")

    return results

# Set image dimensions (width and height)
image_width = 64  # Adjust this based on your image dimensions   
image_height = 64

# Process images and calculate error percentages
error_percentages = process_images(11, image_width, image_height)

# Now plotting the results using matplotlib
# Extract the image names (IMG1, IMG2, ...) and error percentages
images = list(error_percentages.keys())
error_values = list(error_percentages.values())

# Plotting a line plot
plt.figure(figsize=(10, 6))
plt.plot(images, error_values, marker='o', color='skyblue', linestyle='-', linewidth=2, markersize=8)

# Adding labels and title
plt.xlabel('Image')
plt.ylabel('Error Percentage (%)')
plt.title('Error Percentage for Each Image')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
