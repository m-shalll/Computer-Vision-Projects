import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import cv2
from pathlib import Path

# 2-dimensional median smoothing filter for non grey scale images
def median_filter(image, kernel_size):
    # # Create a copy of the image to avoid modifying the original
    # filtered_image = image.copy()
    
    # # Get the dimensions of the image
    # height, width = image.shape
    
    # # Calculate the padding needed for the kernel
    # pad = kernel_size // 2
    
    # # Pad the image with zeros
    # # mode='constant' with constant_values=0 will pad the image with zeros
    # padded_image = np.pad(image, pad_width=pad, mode='constant', constant_values=0)
    
    # # Apply the median filter
    # for i in range(height):
    #     for j in range(width):
    #         # Extract the neighborhood
    #         # The neighborhood is a square region of size kernel_size x kernel_size centered around the current pixel
    #         # center is the pixel at (i, j) in the original image, which corresponds to (i+pad, j+pad) in the padded image
    #         neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size]
    #         # Replace the pixel with the median of the neighborhood
    #         filtered_image[i, j] = np.median(neighborhood)
    
    # Use OpenCV's built-in median filter for better performance and simplicity
    filtered_image = cv2.medianBlur(image, kernel_size)


    return filtered_image


# Canny’s algorithm for edge detection
def canny_edge_detection(grayscale_image, low_threshold=100, high_threshold=200):

    edges = cv2.Canny(grayscale_image, low_threshold, high_threshold)
    return edges

# define a polygon (region) of the image to mask then oise edges producing only an edge image that contains the region of interest that focus on the road.
def region_of_interest(image, vertices, color=(255, 255, 255)):
    # Create a mask with the same dimensions as the image
    mask = np.zeros_like(image)
    
    # Fill the polygon defined by vertices with the specified color
    cv2.fillPoly(mask, [vertices], color)
    
    # Apply the mask to the image using bitwise AND
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image


def hough_transform(edges, rho_resolution=1, theta_resolution=np.pi/180):
    height, width = edges.shape

    # Maximum possible rho (image diagonal)
    diag_len = int(np.ceil(np.sqrt(height**2 + width**2)))
    rhos = np.arange(-diag_len, diag_len + 1, rho_resolution)

    # Theta values
    thetas = np.arange(0, np.pi, theta_resolution)

    # Accumulator array
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

    # Indices of edge points
    y_idxs, x_idxs = np.nonzero(edges)

    # Precompute cos and sin of thetas for efficiency
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Vectorized computation of ρ for all θ for each edge point
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        # Compute ρ for all θ at once using vectorized operations
        rho_vals = x * cos_t + y * sin_t
        # Convert ρ values to corresponding indices in the accumulator
        rho_idxs = np.round((rho_vals + diag_len) / rho_resolution).astype(int)
        accumulator[rho_idxs, np.arange(len(thetas))] += 1

    return accumulator, rhos, thetas

# Function to find peaks in the Hough accumulator
def find_hough_peaks(accumulator, num_peaks, threshold=50):
    # Flatten the accumulator and find indices of the top peaks
    flat_indices = np.argpartition(accumulator.flatten(), -num_peaks)[-num_peaks:]
    peak_indices = np.unravel_index(flat_indices, accumulator.shape)
    
    # Filter peaks based on the threshold
    peaks = []
    for i in range(len(peak_indices[0])):
        rho_idx = peak_indices[0][i]
        theta_idx = peak_indices[1][i]
        if accumulator[rho_idx, theta_idx] > threshold:  # Threshold for peak detection
            peaks.append((rho_idx, theta_idx))
    return peaks

# Function to draw lines corresponding to the detected peaks on the original image
def draw_lines(image, peaks, rhos, thetas):
    for rho_idx, theta_idx in peaks:
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # Calculate the start and end points of the line segment accurately based on the line parameters
        x1 = int(x0 + max(image.shape) * (-b))
        y1 = int(y0 + max(image.shape) * (a))
        x2 = int(x0 - max(image.shape) * (-b))
        y2 = int(y0 - max(image.shape) * (a))
        # Draw the line on the image
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

if __name__ == "__main__":
    # Load the image
    image_path = Path(__file__).resolve().parent / 'part2-image.png'
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")
    
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply median filter
    median_filtered_image = median_filter(grayscale_image, kernel_size=3)
    
    # Apply Canny edge detection
    edges = cv2.Canny(median_filtered_image, 150, 250)

    # Define the vertices of the region of interest (a polygon that focuses on the road)
    height, width = edges.shape
    vertices = np.array([
        (0, height),          # Bottom-left corner
        (0, height // 2 + 100),  # Left-middle point
        (width // 2 - 50, height//2 -40),  # Left-middle point
        (width, height)       # Bottom-right corner
    ], dtype=np.int32)
    # Apply the region of interest mask to the edges image
    masked_edges = region_of_interest(edges, vertices)
    masked_edges_pixels = np.argwhere(masked_edges > 0)

    hough_transform_accumulator, rhos, thetas = hough_transform(masked_edges)
    peaks = find_hough_peaks(hough_transform_accumulator, num_peaks=9)
    black_image = np.zeros_like(image)
    draw_lines(black_image, peaks, rhos, thetas)
    # cv2.imshow('Detected Lines on Black Image', black_image)
    masked_black_image = region_of_interest(black_image, vertices, color=(0, 255, 0))
    # cv2.imshow('Detected Lines on Black Image', masked_black_image)

    
    # Combine the original image with the detected lines
    combined_image = cv2.addWeighted(image, 0.8, masked_black_image, 1, 0)
    #show original image vs original image with detected lines
    cv2.imshow('Original Image', image)
    cv2.imshow('Combined Image with Detected Lines', combined_image)

    # Display the results
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Median Filtered Image', median_filtered_image)
    # cv2.imshow('Canny Edges', edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()