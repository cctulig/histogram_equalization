# algorithm-prototyping
## Auto tester
### app.py
Run app.py to run the auto tester.
This file loads the canvas that all the images and text are displayed on.
How to use the auto tester:
    
    - 'a' to accept an image
    - 'd' to deny an image
    - 'right arrow' to navigate to the next image
    - 'left arrow' to navigate to a previous image

### Scan Controller
Class that holds all the information of all the image processing results

### Helper Functions
Functions for resizing the images as well as running the edge detection algorithms

IMPORTANT: Change line 27 to change which histogram equalization algorithm is being used.

## Histogram Equalization Algorithms:
### CLAHE Method
This is the state-of-the-art histogram equalization method for grayscale images.

Source: https://docs.opencv.org/3.4/d5/daf/tutorial_py_histogram_equalization.html

### RGB Histogram Equalization
Another method for histogram equalization that instead balanced out each of the red, green, and blue channels.

Source: https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image

# Insert Folder Name Here
## Insert Subsection Here
### Insert Item Title Here
Insert item description here.
