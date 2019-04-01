# Compute absolute colour difference of two images.
# The two images must have the same size.
# Return combined absolute difference of the 3 channels 

def absDiff(image1, image2):
	if image1.shape != image2.shape:
		print "image size mismatch"
		return 0
	else:
		height,width,dummy = image1.shape
		# Compute absolute difference.
		diff = cv2.absdiff(image1, image2)
		a = cv2.split(diff)
		# Sum up the differences of the 3 channels with equal weights.
		# You can change the weights to different values.
		sum = np.zeros((height,width), dtype=np.uint8)
		for i in (1, 2, 3):
			ch = a[i-1]
			cv2.addWeighted(ch, 1.0/i, sum, float(i-1)/i, gamma=0.0, dst=sum)
		return sum
		
	
#-----------------------------
# -----------------------------
# Main

# Initialisation
import cv2
import numpy as np

# Initialisation
filename1 = "eagle-1.jpg"
filename2 = "eagle-2.jpg"
difffilename = "eagle-diff.jpg"

# Load images
image1 = cv2.imread(filename1)
image2 = cv2.imread(filename2)

# Compute colour difference and remove background
diff = absDiff(image1, image2)
cv2.imwrite(difffilename, diff)
