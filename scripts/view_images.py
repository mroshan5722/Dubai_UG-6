import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure

# Load the image array
img_array = np.load('data/raw_data/cloudcast/CloudCastSmall/TrainCloud/15747.npy')

# Increase contrast using contrast stretching
p2, p98 = np.percentile(img_array, (2, 98))
img_rescale = exposure.rescale_intensity(img_array, in_range=(p2, p98))

# Display the enhanced image
plt.imshow(img_rescale, cmap='gray')

# Save the enhanced image to a file
plt.savefig('cloud_image_contrast.png')
print("Image saved as 'cloud_image_contrast.png'")
