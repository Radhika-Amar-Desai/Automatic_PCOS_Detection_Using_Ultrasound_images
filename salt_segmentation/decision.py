import cv2
import numpy as np
from segmentation import final_func as salt_segmentation
from segmentation import erode_image, dilate_image
from segmentation import adaptive_histogram_equalization

def mark_speckle( gray_image ):

    filtered_image = cv2.medianBlur( gray_image , 5 )
    speckle_noise = cv2.absdiff(gray_image, filtered_image)
    _, binary_image = cv2.threshold(speckle_noise, 30, 255, cv2.THRESH_BINARY)

    image = cv2.cvtColor ( gray_image, cv2.COLOR_GRAY2BGR )

    marked_image = image.copy()
    marked_image[binary_image > 0] = [0, 0, 255]  # Mark in red (BGR color format)

    return marked_image


def filter_speckle ( image , box_size = 5 , iterations = 1 ):
    height, width, _ = image.shape

    half_box = box_size // 2

    for _ in range ( iterations ):
        for y in range(height):
            for x in range(width):
                # Check if the current pixel is red ([0, 0, 255])
                if np.array_equal(image[y, x], [0, 0, 255]):
                    # Define the coordinates of the box neighborhood
                    x_min = max(0, x - half_box)
                    x_max = min(width, x + half_box + 1)
                    y_min = max(0, y - half_box)
                    y_max = min(height, y + half_box + 1)

                    # Check if there is a black pixel ([0, 0, 0]) in the neighborhood
                    has_black_pixel = False
                    for ny in range(y_min, y_max):
                        for nx in range(x_min, x_max):
                            if np.array_equal(image[ny, nx], [0, 0, 0]):
                                has_black_pixel = True
                                break
                        if has_black_pixel:
                            break

                    # If there is a black pixel in the neighborhood, change the red pixel to black
                    if has_black_pixel:
                        image[y, x] = [0, 0, 0]

    return image

def speckle_noise( image ):
    
    return len( [ pixel for row in image for pixel in row if list( pixel ) == [ 0 , 0 , 255 ]] ) 

def preprocess_image ( image ):
    if ( speckle_noise ( filter_speckle ( mark_speckle ( image ) ) ) <= 1000 ):
        return erode_image ( salt_segmentation ( image ) )
    else:
        return salt_segmentation ( image )
