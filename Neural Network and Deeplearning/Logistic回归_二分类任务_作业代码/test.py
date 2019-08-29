import os
from LogisticsP import *
import skimage.transform as tsf

print("\n\n--------test on my own image--------\n")
num_px = 64
my_image = "cat1.jpg"
fname = os.path.join(os.getcwd(), "images")
picpath = os.path.join(fname, my_image)

image = np.array(plt.imread(picpath))
#print(image.shape)
my_image = tsf.resize(image, (num_px, num_px), mode='constant').reshape((1, num_px*num_px*3)).T
#print(my_image.shape)
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
