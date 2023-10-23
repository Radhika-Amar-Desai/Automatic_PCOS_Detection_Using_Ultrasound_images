import cv2
import numpy as np

# Load the binary image (make sure it's already thresholded)
image = cv2.imread('Processed_images/infected/img_3.jpg', cv2.IMREAD_GRAYSCALE)
image = 255 - image

# Smoothen the image with Gaussian blur
smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)

# Sharpen the image using unsharp masking
sharpened_image = cv2.addWeighted(image, 1.5, smoothed_image, -0.5, 0)

# Find contours in the sharpened image
contours, _ = cv2.findContours(sharpened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

object_count = 0
object_areas = []
object_perimeters = []

# Create a copy of the original image to draw the selected contours
original_with_contours = cv2.cvtColor(sharpened_image, cv2.COLOR_GRAY2BGR)

for contour in contours:
    # Calculate area for each object
    area = cv2.contourArea(contour)

    # Calculate perimeter for each object
    perimeter = cv2.arcLength(contour, closed=True)

    # Filter out small noise by setting a threshold on area
    if 10 < area :  # Adjust the threshold as needed
        # Calculate other shape characteristics
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)

        # Only draw contours that are not straight lines (num_vertices > 2)
        if num_vertices >= 5:
            object_count += 1
            object_areas.append(area)
            object_perimeters.append(perimeter)

            # Draw the contour on the copy of the original image
            cv2.drawContours(original_with_contours, [contour], -1, (0, 0, 255), 1)  # You can change the color and thickness

print("Number of follicles:", object_count)
for i in range(object_count):
    print(f"Object {i + 1} - Area: {object_areas[i]}, Perimeter: {object_perimeters[i]}")

# Display the original image with selected contours
cv2.imshow ('Original Imgae', sharpened_image)
cv2.imshow('Original Image with Contours', original_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
