import matplotlib.pyplot as plt
import numpy as np

# Read and parse the data from data.txt
error_data = []
with open("data.txt", "r") as file:
    current_try = []
    for line in file:
        if line.startswith("Error Percentage for IMG"):
            # Extract the percentage
            error_percentage = float(line.split(": ")[1].strip().replace("%", ""))
            current_try.append(error_percentage)
        elif line.strip() == "" and current_try:
            # Ensure each try has exactly 11 images before adding it to error_data
            if len(current_try) == 11:
                error_data.append(current_try)
            current_try = []
    if current_try and len(current_try) == 11:  # Add the last try if it has 11 images
        error_data.append(current_try)

# Convert to a numpy array for easier manipulation
error_data = np.array(error_data)

# Calculate the average error percentage for each image
average_errors = error_data.mean(axis=0)

# Print the average error for each image
print("Average Error Percentage for each image:")
for i, avg_error in enumerate(average_errors, 1):
    print(f"IMG{i}: {avg_error:.2f}%")

# Plot error percentages for each try
plt.figure(figsize=(12, 6))
for i, errors in enumerate(error_data, 1):
    plt.plot(range(1, 12), errors, label=f'Try {i}', marker='o')
plt.title("Error Percentage for Each Try")
plt.xlabel("Image Number")
plt.ylabel("Error Percentage (%)")
plt.legend()
plt.show()

# Plot the average error percentage for each image
plt.figure(figsize=(12, 6))
plt.plot(range(1, 12), average_errors, marker='o', color='b', label='Average Error')
plt.title("Average Error Percentage per Image")
plt.xlabel("Image Number")
plt.ylabel("Average Error Percentage (%)")
plt.legend()
plt.show()
