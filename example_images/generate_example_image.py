import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image



temp_img = Image.open('image.png')
image = temp_img.resize((640, 384))

figure, ax = plt.subplots()

ax.imshow(image)

# forward box [250:391, 250:321]
forward = patches.Rectangle((250, 250), 140, 70, linewidth=2, edgecolor='red', facecolor='none', label='forward')

# rotate box [160:481, 270:311]
rotate = patches.Rectangle((160, 270), 320, 40, linewidth=2, edgecolor='blue', facecolor='none', label='rotate')

# right box [440:641, 100:281]
right = patches.Rectangle((440, 100), 200, 180, linewidth=2, edgecolor='darkgoldenrod', facecolor='none', label='right')

# left box [0:201, 100:281]
left = patches.Rectangle((1, 100), 200, 180, linewidth=2, edgecolor='mediumorchid', facecolor='none', label='left')


ax.add_patch(forward)
ax.add_patch(rotate)
ax.add_patch(right)
ax.add_patch(left)

# plt.legend(loc='upper right')

# plt.show()

plt.savefig('example_image.png')