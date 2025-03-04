import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from scipy.optimize import differential_evolution

###############################
# 1) IMAGE PROCESSING PIPELINE
###############################

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Step 1: Convert Color Image to Greyscale
def convert_to_greyscale(image_number, output_dir):
    sample_images_folder = "Sample Images"  # Folder with TIFF images (e.g., Sample_1.tiff)
    tiff_image_path = os.path.join(sample_images_folder, f'Sample_{image_number}.tiff')
    grey_image_path = os.path.join(output_dir, 'greyscale_image.png')
    img = Image.open(tiff_image_path)
    grey_img = img.convert("L")  # Convert to greyscale
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
    images_folder = "Images"  # Folder with color images (e.g., IMG1.png)
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
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_error_percentage(red_pixels_file, boundary_pixels_file, image_width, image_height):
    red_pixels = read_and_sort_coordinates(red_pixels_file)
    boundary_pixels = read_and_sort_coordinates(boundary_pixels_file)
    max_distance = math.sqrt(image_width**2 + image_height**2)
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

###############################
# 2) SIGMOID TRANSFORM
###############################
def apply_sigmoid_transform(image_number, input_dir, output_dir):
    input_image_path = os.path.join(input_dir, 'greyscale_image.png')
    img = Image.open(input_image_path).convert("L")
    img_arr = np.array(img, dtype=np.float32)
    
    # Normalize to [-1, 1] and apply sigmoid transformation.
    img_arr = (img_arr - 128) / 128
    transformed_arr = 1 / (1 + np.exp(-img_arr * 8))  # scaling factor of 8
    transformed_uint8 = (transformed_arr * 255).astype(np.uint8)
    
    transformed_image_path = os.path.join(output_dir, f"transformed_image_{image_number}.png")
    Image.fromarray(transformed_uint8, mode="L").save(transformed_image_path)
    print(f"Transformed image saved: {transformed_image_path}")
    
    # Overlay 100 random points for illustration.
    width, height = img.size
    xs = np.random.randint(0, width, size=100)
    ys = np.random.randint(0, height, size=100)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(transformed_uint8, cmap='gray', origin='upper')
    plt.scatter(xs, ys, color='red', marker='o', s=30)
    plt.title(f"Transformed Image with 100 Random Points - IMG{image_number}")
    plt.axis('off')
    
    overlaid_image_path = os.path.join(output_dir, f"transformed_image_with_points_{image_number}.png")
    plt.savefig(overlaid_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return transformed_image_path

###############################
# 3) FIREFLY ANIMATION (Corrected)
###############################
def read_coordinates(file_path):
    """Read coordinates from a file written as (x, y) on each line."""
    coords = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip("()\n").split(",")
            if len(parts) == 2:
                coords.append((int(parts[0]), int(parts[1])))
    return np.array(coords)

def animate_points_firefly_towards_boundary(
    image_path,
    boundary_file,
    num_points=100,
    beta0=2.0,   # Increased base attractiveness for stronger pull
    gamma=0.1,   # Reduced absorption so even distant points feel attraction
    alpha=0.01,  # Very small random component
    frames=100   # More frames for gradual convergence
):
    """
    Animate points moving toward the nearest boundary pixel using the Firefly update:
    
        S_i = S_i + β0 * exp(-γ * r_ij^2) * (S_j - S_i) + α * ε_i
    """
    # Load image and get its dimensions.
    img = Image.open(image_path).convert("L")
    img_arr = np.array(img)
    height, width = img_arr.shape

    # Read boundary coordinates.
    boundary_coords = read_coordinates(boundary_file)
    if boundary_coords.size == 0:
        print("No boundary pixels found! The points have nowhere to go.")
        return

    # Initialize random firefly positions.
    np.random.seed(42)
    xs = np.random.randint(0, width, size=num_points)
    ys = np.random.randint(0, height, size=num_points)
    # Ensure positions is shape (num_points, 2)
    positions = np.vstack((xs, ys)).T.astype(float)

    # Store positions for each frame.
    all_positions = [positions.copy()]

    for _ in range(frames - 1):
        new_positions = positions.copy()
        for i in range(num_points):
            # Find the nearest boundary pixel.
            diff = boundary_coords - positions[i]  # positions[i] is shape (2,)
            dist = np.sqrt(np.sum(diff**2, axis=1))
            j_idx = np.argmin(dist)
            S_j = boundary_coords[j_idx]
            r_ij = dist[j_idx]

            # Update using Firefly rule.
            beta = beta0 * np.exp(-gamma * (r_ij**2))
            epsilon = np.random.uniform(-1, 1, size=2)
            new_positions[i] = positions[i] + beta * (S_j - positions[i]) + alpha * epsilon

            # Ensure positions remain within image bounds.
            new_positions[i, 0] = np.clip(new_positions[i, 0], 0, width - 1)
            new_positions[i, 1] = np.clip(new_positions[i, 1], 0, height - 1)

        positions = new_positions
        all_positions.append(positions.copy())

    # Animate the convergence.
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_arr, cmap='gray', origin='upper')
    scat = ax.scatter(all_positions[0][:, 0], all_positions[0][:, 1], color='red', s=30)
    ax.set_title("Firefly Algorithm: Points Moving Toward Boundary")
    ax.axis('off')

    def update(frame):
        scat.set_offsets(all_positions[frame])
        return (scat,)

    anim = FuncAnimation(fig, update, frames=frames, interval=200, blit=True, repeat=False)
    plt.show()

    # To save the animation as a GIF, uncomment these lines:
    # anim.save("firefly_animation.gif", writer='pillow', fps=5)
    # print("Animation saved as firefly_animation.gif")

###############################
# 4) OPTIONAL: OPTIMIZATION TEST FUNCTIONS
###############################
def rastrigin(x):
    A = 10
    return A * 2 + (x[0]**2 - A * np.cos(2*np.pi*x[0])) + (x[1]**2 - A * np.cos(2*np.pi*x[1]))

def ackley(x):
    a, b, c = 20, 0.2, 2*np.pi
    sum_sq = x[0]**2 + x[1]**2
    sum_cos = np.cos(c*x[0]) + np.cos(c*x[1])
    return -a*np.exp(-b*np.sqrt(sum_sq/2)) - np.exp(sum_cos/2) + a + np.e

def sphere(x):
    return x[0]**2 + x[1]**2

def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def beale(x):
    x1, x2 = x
    return (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2

def goldstein_price(x):
    x1, x2 = x
    term1 = (x1 + x2 + 1)**2
    term2 = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
    term3 = (2*x1 - 3*x2)**2
    term4 = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
    return (1 + term1*term2) * (30 + term3*term4)

def booth(x):
    x1, x2 = x
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

def bukin(x):
    x1, x2 = x
    return 100*np.sqrt(abs(x2 - 0.01*x1**2)) + 0.01*abs(x1 + 10)

def matyas(x):
    x1, x2 = x
    return 0.26*(x1**2 + x2**2) - 0.48*x1*x2

def levi(x):
    x1, x2 = x
    return np.sin(3*np.pi*x1)**2 + ((x1 - 1)**2)*(1+np.sin(3*np.pi*x2)**2) + ((x2 - 1)**2)*(1+np.sin(2*np.pi*x2)**2)

def griewank(x):
    sum_sq = (x[0]**2 + x[1]**2) / 4000.0
    prod_cos = np.cos(x[0]) * np.cos(x[1]/np.sqrt(2))
    return 1 + sum_sq - prod_cos

def himmelblau(x):
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

def three_hump_camel(x):
    x1, x2 = x
    return 2*x1**2 - 1.05*x1**4 + (x1**6)/6 + x1*x2 + x2**2

def easom(x):
    x1, x2 = x
    return -np.cos(x1)*np.cos(x2)*np.exp(-((x1-np.pi)**2+(x2-np.pi)**2))

def cross_in_tray(x):
    x1, x2 = x
    return -0.0001*(abs(np.sin(x1)*np.sin(x2)*np.exp(abs(100 - np.sqrt(x1**2+x2**2)/np.pi)))+1)**0.1

def eggholder(x):
    x1, x2 = x
    return -(x2+47)*np.sin(np.sqrt(abs(x1/2+(x2+47)))) - x1*np.sin(np.sqrt(abs(x1-(x2+47))))

def holder_table(x):
    x1, x2 = x
    return -abs(np.sin(x1)*np.cos(x2)*np.exp(abs(1-np.sqrt(x1**2+x2**2)/np.pi)))

def mccormick(x):
    x1, x2 = x
    return np.sin(x1+x2) + (x1-x2)**2 - 1.5*x1 + 2.5*x2 + 1

def schaffer_N2(x):
    x1, x2 = x
    num = (np.sin(x1**2 - x2**2))**2 - 0.5
    den = (1+0.001*(x1**2+x2**2))**2
    return 0.5 + num/den

def schaffer_N4(x):
    x1, x2 = x
    num = np.cos(np.sin(abs(x1**2 - x2**2)))
    den = (1+0.001*(x1**2+x2**2))**2
    return 0.5 + (num - 0.5)/den

def styblinski_tang(x):
    return 0.5 * ((x[0]**4 - 16*x[0]**2 + 5*x[0]) + (x[1]**4 - 16*x[1]**2 + 5*x[1]))

def shekel(x):
    a = np.array([[4,4],[1,1],[8,8],[6,6],[3,7],[2,9],[5,5],[8,1],[6,2],[7,3]])
    c = np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5])
    sum_term = 0
    for i in range(10):
        inner = (x[0]-a[i,0])**2 + (x[1]-a[i,1])**2
        sum_term += 1.0/(inner+c[i])
    return sum_term

def eggcrate(x):
    x1, x2 = x
    return x1**2 + x2**2 + 25*(np.sin(x1)**2 + np.sin(x2)**2)

def drop_wave(x):
    x1, x2 = x
    r = np.sqrt(x1**2+x2**2)
    return -(1+np.cos(12*np.pi*r))/(0.5*r**2+2)

def schaffer_N1(x):
    x1, x2 = x
    return 0.5 + (np.sin(x1**2 - x2**2)**2 - 0.5)/(1+0.001*(x1**2+x2**2))**2

def zakharov(x):
    x1, x2 = x
    s = 0.5*x1 + x2
    return x1**2 + x2**2 + s**2 + s**4

def michalewicz(x, m=10):
    x1, x2 = x
    return -(np.sin(x1)*(np.sin(x1**2/np.pi))**(2*m) + np.sin(x2)*(np.sin(2*x2**2/np.pi))**(2*m))

def chichinadze(x):
    x1, x2 = x
    return x1**2 + x2**2 - 0.3*np.cos(3*np.pi*x1) - 0.4*np.cos(4*np.pi*x2) + 0.7

def dixon_price(x):
    x1, x2 = x
    return (x1 - 1)**2 + 2*(2*x2**2 - x1)**2

def perm_function(x, beta=0.5):
    x1, x2 = x
    S1 = (1+beta)*(x1 - 1) + (2+beta)*((x2/2) - 1)
    S2 = (1+beta)*(x1**2 - 1) + (4+beta)*(((x2/2)**2) - 1)
    return S1**2 + S2**2

functions = {
    "Rastrigin":       (rastrigin,       [(-5.12, 5.12), (-5.12, 5.12)]),
    "Ackley":          (ackley,          [(-5, 5), (-5, 5)]),
    "Sphere":          (sphere,          [(-5.12, 5.12), (-5.12, 5.12)]),
    "Rosenbrock":      (rosenbrock,      [(-5, 5), (-5, 5)]),
    "Beale":           (beale,           [(-4.5, 4.5), (-4.5, 4.5)]),
    "GoldsteinPrice":  (goldstein_price, [(-2, 2), (-2, 2)]),
    "Booth":           (booth,           [(-10, 10), (-10, 10)]),
    "Bukin":           (bukin,           [(-15, -5), (-3, 3)]),
    "Matyas":          (matyas,          [(-10, 10), (-10, 10)]),
    "Levi":            (levi,            [(-10, 10), (-10, 10)]),
    "Griewank":        (griewank,        [(-600, 600), (-600, 600)]),
    "Himmelblau":      (himmelblau,      [(-5, 5), (-5, 5)]),
    "ThreeHumpCamel":  (three_hump_camel,[(-5, 5), (-5, 5)]),
    "Easom":           (easom,           [(-100, 100), (-100, 100)]),
    "CrossInTray":     (cross_in_tray,   [(-10, 10), (-10, 10)]),
    "Eggholder":       (eggholder,       [(-512, 512), (-512, 512)]),
    "HolderTable":     (holder_table,    [(-10, 10), (-10, 10)]),
    "McCormick":       (mccormick,       [(-1.5, 4), (-3, 4)]),
    "SchafferN2":      (schaffer_N2,     [(-100, 100), (-100, 100)]),
    "SchafferN4":      (schaffer_N4,     [(-100, 100), (-100, 100)]),
    "StyblinskiTang":  (styblinski_tang, [(-5, 5), (-5, 5)]),
    "Shekel":          (shekel,          [(-5, 10), (-5, 10)]),
    "Eggcrate":        (eggcrate,        [(-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi)]),
    "DropWave":        (drop_wave,       [(-5.12, 5.12), (-5.12, 5.12)]),
    "SchafferN1":      (schaffer_N1,     [(-100, 100), (-100, 100)]),
    "Zakharov":        (zakharov,        [(-5, 5), (-5, 5)]),
    "Michalewicz":     (michalewicz,     [(0, np.pi), (0, np.pi)]),
    "Chichinadze":     (chichinadze,     [(-10, 10), (-10, 10)]),
    "DixonPrice":      (dixon_price,     [(-10, 10), (-10, 10)]),
    "Perm":            (perm_function,   [(-2, 2), (-2, 2)])
}

def process_optimization_functions():
    output_folder = "Optimization_Results"
    ensure_directory_exists(output_folder)
    results_summary = []
    for fname, (func, bounds) in functions.items():
        res_min = differential_evolution(func, bounds, disp=False)
        min_val = res_min.fun
        min_x = res_min.x
        
        neg_func = lambda x: -func(x)
        res_max = differential_evolution(neg_func, bounds, disp=False)
        max_val = -res_max.fun
        max_x = res_max.x
        
        file_path = os.path.join(output_folder, f"{fname}_results.txt")
        with open(file_path, "w") as f:
            f.write(f"Function: {fname}\n")
            f.write(f"Bounds: {bounds}\n\n")
            f.write(f"Global Minimum value: {min_val:.6f}\n")
            f.write(f"Found at x = {min_x}\n\n")
            f.write(f"Global Maximum value: {max_val:.6f}\n")
            f.write(f"Found at x = {max_x}\n")
        
        results_summary.append((fname, min_val, min_x, max_val, max_x))
        print(f"Processed {fname}")
    
    print("\nSummary of Optimization Results:")
    for item in results_summary:
        fname, min_val, min_x, max_val, max_x = item
        print(f"{fname}: min {min_val:.6f} at {min_x}, max {max_val:.6f} at {max_x}")

###############################
# MAIN FUNCTION
###############################
def main():
    num_images = 11
    image_width = 64
    image_height = 64

    # Process images: convert to greyscale, extract gray markings, detect boundaries, and extract red pixels.
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

    # Apply sigmoid transform and save transformed images.
    transformed_output_dir = "Transformed_Images"
    ensure_directory_exists(transformed_output_dir)
    
    for i in range(1, num_images + 1):
        input_dir = f'Image_{i}'
        transformed_image_path = apply_sigmoid_transform(i, input_dir, transformed_output_dir)
        print(f"Transformed image for IMG{i} saved.")
        
        # For demonstration, run the Firefly animation only for image 1.
        if i == 1:
            boundary_file = os.path.join(input_dir, "boundary_pixels.txt")
            print(f"\nAnimating points (Firefly) moving toward the boundary for IMG{1}...")
            animate_points_firefly_towards_boundary(
                image_path=transformed_image_path,
                boundary_file=boundary_file,
                num_points=100,
                beta0=2.0,
                gamma=0.1,
                alpha=0.01,
                frames=100
            )

    # Run optimization test functions (optional).
    print("\nStarting optimization on test functions...")
    process_optimization_functions()

if __name__ == "__main__":
    main()
