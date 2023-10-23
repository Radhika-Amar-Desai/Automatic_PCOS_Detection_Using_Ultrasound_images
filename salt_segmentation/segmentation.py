import cv2
import numpy as np

def resize_image ( image ):
    return cv2.resize ( image , ( 250,250 ) )

def adaptive_histogram_equalization(gray_image, clip_limit=2.0, grid_size=(8, 8)):
    
    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    equalized_image = clahe.apply(gray_image)

    equalized_bgr = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

    return equalized_bgr

def erode_image( binary_image : list , ksize = 3 ):
    kernel = np.ones((ksize, ksize), np.uint8)
    if binary_image is not None : return cv2.dilate(binary_image, kernel, iterations=1)

def dilate_image( binary_image : list, ksize = 3 ):
    kernel = np.ones((ksize, ksize), np.uint8)
    if binary_image is not None : return cv2.erode(binary_image, kernel, iterations=1)

def create_new_canvas ( boxes, gray_image ):
    canvas = np.ones_like(gray_image) * 255
    for box in boxes:
        x, y, width, height = box
        if ( width > 8 and height > 8 ):
            cropped_region = gray_image[ y - height : y + height + 1 , x - width : x + width + 1 ]
            canvas [ y - height : y + height + 1 , x - width : x + width + 1]\
                = cropped_region
    return canvas

def calculate_otsu_threshold(gray_image):

    gray_image = gray_image[gray_image != 255]

    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    histogram /= histogram.sum()

    optimal_threshold = -1
    max_variance = 0

    for threshold in range(255):

        p1 = histogram[:threshold].sum()
        p2 = histogram[threshold:].sum()

        m1 = np.sum(np.arange(0, threshold) * histogram[:threshold]) / (p1 if p1 > 0 else 1)
        m2 = np.sum(np.arange(threshold, 256) * histogram[threshold:]) / (p2 if p2 > 0 else 1)

        between_class_variance = p1 * p2 * (m1 - m2) ** 2

        if between_class_variance > max_variance:
            max_variance = between_class_variance
            optimal_threshold = threshold

    return optimal_threshold

def apply_bin_threshold_canvas(gray_canvas):
    
    return np.where( gray_canvas > calculate_otsu_threshold( gray_canvas ), 1, 0).astype(np.uint8)

boxes = []

def display_image_with_red_dots( gray_image , lower_threshold, var_threshold ):
        
    def draw_box(image, y, x, size, color, thickness):
        if ( size != 1 ):
            size += 7
            cv2.rectangle(image, (x - size, y - size), (x + size, y + size), color, thickness)

    def redraw_dots():
        image = cv2.cvtColor ( gray_image, cv2.COLOR_GRAY2BGR )
        image = adaptive_histogram_equalization(image)
        img_with_dots = image.copy()

        for _ in range(red_dots_count):
            idx = np.random.choice(len(dark_pixels[0]))
            y, x = dark_pixels[0][idx], dark_pixels[1][idx]
            img_with_dots[y:y+2, x:x+2] = [0, 0, 255]  

            box_size = 1
            while lower_threshold <= np.var(gray_image[y-box_size:y+box_size+1, x-box_size:x+box_size+1])\
                      < var_threshold:
                
                box_size += 1
                if y - box_size < 0 or y + box_size >= gray_image.shape[1] or x - box_size < 0 or x + box_size >= gray_image.shape[0]:
                    break

            draw_box(img_with_dots, y, x, size=box_size, color=(0, 255, 0), thickness=1)

            boxes.append((x,y,box_size+7,box_size+7))

        #cv2.imshow('Image', img_with_dots)

    #cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Image', 800, 600)

    gray_image = cv2.cvtColor ( gray_image, cv2.COLOR_BGR2GRAY )
    _, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_pixels = np.where(thresholded == 0)

    red_dots_count = 3000
    redraw_dots()

def final_func(gray_image):

    gray_image = adaptive_histogram_equalization( resize_image ( gray_image) )

    display_image_with_red_dots(gray_image , lower_threshold = 0, var_threshold = 100 )
    canvas = create_new_canvas(boxes,gray_image)
    #cv2.imshow("Canvas",canvas)

    final_image = apply_bin_threshold_canvas(canvas) * 255
    # final_image = erode_image(final_image )

    #cv2.imshow("Original Image", gray_image)
    #cv2.imshow("Processed Canvas",final_image) 
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return final_image

image_path = "PCOS_ultrasonic/infected/img6.jpg"
gray_image = cv2.imread ( image_path , 0 )
processed_img = final_func ( gray_image )
cv2.imshow ( "Result" , processed_img )
cv2.waitKey ( 0 )