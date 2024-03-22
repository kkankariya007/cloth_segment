import cv2
import torch

# Load the image
image = cv2.imread('output\\Gen fill_seg\\1.png')

def ford(self, x):

    hx = x

    hxin = self.rebnconvin(hx)

    hx1 = self.rebnconv1(hxin)
    hx2 = self.rebnconv2(hx1)
    hx3 = self.rebnconv3(hx2)

    hx4 = self.rebnconv4(hx3)

    hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
    hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
    hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

    return hx1d + hxin


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.imshow('Image', contours)
# cv2.waitKey(0)

max_width = 0

# Iterate through contours
for contour in contours:
    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)
    # Update max_width if the width of the contour is greater
    if w > max_width:
        max_width = w

# Display the maximum width
# print("The width of the widest white region is:", max_width)
if max_width>1100 and max_width<1200:
    print("Recommended shirt size is S")
elif max_width>1200 and max_width<1300:
    print("Recommended shirt size is M")
elif max_width>1200 and max_width<1300:
    print("Recommended shirt size is L")
elif max_width>1300 and max_width<1600:
    print("Recommended shirt size is XL")
else:
    print("Recommeded shirt size is XXL and above")
