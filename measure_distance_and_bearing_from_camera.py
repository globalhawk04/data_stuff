# figuring out how to located these poles in lat and long.
# i know i need to figure out how to estimate the distance.
# also need to estimate angle 

from imutils import paths
import numpy as np
import imutils
import cv2
import math

import PIL
from PIL import Image
 

top_right = []
top_left = []

def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)
	# compute the bounding box of the of the paper region and return it
	#print(cv2.minAreaRect(c))
	z = cv2.minAreaRect(c)
	top_right.append(tuple(z[0]))
	top_left.append(tuple(z[1]))
	#print(z[0])
	#print(z[1])

	return cv2.minAreaRect(c)




def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth
	#print((knownWidth * focalLength) / perWidth)

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 72.0
# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 15.0
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread("testing_know.jpg")
marker = find_marker(image)

print('xxxxxxxx')

print(marker)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH


x1, y1, x2, y2 = (450, 450, 2500, 2500)
#x1, y1, x2, y2 = (39, 117, 39+79, 79+270)
#x1, y1, x2, y2 = (300, 195, 320+27, 195+69)
#x1, y1, x2, y2 = (236, 156, 320, 320)
image = PIL.Image.open("testing_know.jpg")
# Crop the image
image.show()
cropped_image = image.crop((x1, y1, x2, y2))
# Save the cropped image
#cropped_image.save("cropped_image.jpg")
cropped_image.show()


# loop over the images
for imagePath in range(1,2):
	# load the image, find the marker in the image, then compute the
	# distance to the marker from the camera
	image = cv2.imread('testing.jpg')
	marker = find_marker(image)
	inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
	print(inches/12)
	# draw a bounding box around the image and display it
	#box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
	#box = np.intp(box)
	#cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	#cv2.putText(image, "%.2fft" % (inches / 12),
		#(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		#2.0, (0, 255, 0), 3)
	#cv2.imshow("image", image)
	#cv2.waitKey(0)


import cv2

# Load the image
image = cv2.imread("testing_know.jpg")

# Get the height and width of the image
height, width = image.shape[:2]

# Find the center of the image
#center = (width // 2, height // 2)
center = (width // 2, 1)



# Print the coordinates of the center pixel
#print(center)

print(top_left)

xone = center[0]
yone = center[1]
#xtwo = top_right[0][0]
#ytwo = top_right[0][1]

#xtwo = top_right[1][0]
#ytwo = top_right[1][1]


xtwo = top_left[0][0]
ytwo = top_left[0][1]

#xtwo = top_left[1][0]
#ytwo = top_left[1][1]


slope = (ytwo-yone)/(xtwo-xone)

print(slope)


def calculate_bearing(ax, ay, bx, by):
    """Computes the bearing in degrees from the point A(a1,a2) to
    the point B(b1,b2). Note that A and B are given in terms of
    screen coordinates.

    Args:
        ax (int): The x-coordinate of the first point defining a line.
        ay (int): The y-coordinate of the first point defining a line.
        bx (int): The x-coordinate of the second point defining a line.
        by (int): The y-coordinate of the second point defining a line.

    Returns:
        float: bearing in degrees
    """
    
    TWO_PI = math.pi * 2
    # if (a1 = b1 and a2 = b2) throw an error 
    theta = math.atan2(bx - ax, ay - by)
    if (theta < 0.0):
        theta += TWO_PI

    print(math.degrees(theta))
    return math.degrees(theta) 

calculate_bearing(xone, yone, xtwo, ytwo)


#3ft
        #86.47990038244266
        #80.55539143279174
        #73.44756277695717
        #72.2629246676514

#4ft
      	#78.1861152141802,
        #52.71639586410635,
        #99.5096011816839,
        #87.66322008862629

#2ft
        #42.646971935007386,
        #35.53914327917282,
        #144.5258493353028,
        #115.50221565731167
