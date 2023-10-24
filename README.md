## Automatic PCOS Detection Using Ultrasound Images

# Overview

The project provides python code for detection system detects if a patient is infected with PCOS based on the ultrasound image of an ovary, helps count number of follicles as well as reveals geometric parameters about the follicles by effectively using a novel segmentation technique called "salt segmentation"

# Installation
Install the pre-requistes using :
```
pip install -r requirements.txt
```

Clone the Github repository and run each file in ML models folder to test the performance of ML models on Images processed by salt segmentation.

Example :
To test the performance of KNN on the processed images we run the following code :
```
python3 ML_Models/KNN.py 
```

# Documentation

The research paper describes the working of "salt segmentation technique" used in the project.

https://docs.google.com/document/d/1Ktj4RPHEFC0uDy_obg5LpBPKFnNUsRYp10MSsLH-Zaw/edit?usp=sharing
