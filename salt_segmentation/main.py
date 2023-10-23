import cv2
import os
import shutil
from decision import preprocess_image

source_folder = 'PCOS_ultrasonic/notinfected'
destination_folder = 'Processed_images/notinfected'

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

files = os.listdir(source_folder)

print(len(files))

file_count = 0

for file in files:

    image_path = source_folder + "/" + file
    image = cv2.imread ( image_path, 0 )
    image = cv2.resize ( image , ( 250,250 ) )

    pre_processed_image = preprocess_image (  image )

    cv2.imwrite ( f"Processed_images/notinfected/img_{file_count}.jpg", pre_processed_image )
    file_count += 1