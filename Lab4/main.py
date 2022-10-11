import skimage  
import os  
import matplotlib.pyplot as plt 
from skimage.color import rgb2gray
from skimage.transform import rescale, resize
from skimage import io
from skimage.exposure import histogram
from skimage.util import img_as_ubyte
from skimage import data
from skimage.feature import match_template
import numpy as np

 

#Task 1
filename_path = "C:/Users/thed7f/Desktop/LTU/Programming for AI/Programming-for-ai-labs/Lab4/"
filename_coins = os.path.join(filename_path,'coins.jpg')
filename_astronaut = os.path.join(filename_path,'astronaut.jpg')  

coins = io.imread(filename_coins)
astronaut = io.imread(filename_astronaut)

print("Task1")

print(coins.shape)
print(coins.dtype)
print(coins[1,1])

print(astronaut.shape)
print(astronaut.dtype)
print(astronaut[1,1])

ax = plt.imshow(coins)
plt.show()
ax = plt.imshow(astronaut)
plt.show()

#Task 2
coins_grayscale = coins
astronaut_grayscale = rgb2gray(astronaut)

print("\nTask2")
print(coins_grayscale.dtype)
print(coins[1,1])

print(astronaut_grayscale.dtype)
print(astronaut_grayscale[1,1])

ax = plt.imshow(coins_grayscale)
plt.show()
ax = plt.imshow(astronaut_grayscale)
plt.show()

#Task 3
print("\nTask3")
print(astronaut_grayscale.shape)
#resizing grayscaled astronaut
image_resized = resize(astronaut_grayscale, (astronaut_grayscale.shape[0] // 2, astronaut_grayscale.shape[1]// 2), 
anti_aliasing=True)
print(image_resized.shape)
print(image_resized.dtype) # float64

#rescaling grayscaled astronaut
image_rescaled_75 = rescale(astronaut_grayscale, 0.75, anti_aliasing=False)
print(image_rescaled_75.shape)
print(image_resized.dtype) # float64

image_rescaled_50 = rescale(astronaut_grayscale, 0.5, anti_aliasing=False)
print(image_rescaled_50.shape)

image_rescaled_25 = rescale(astronaut_grayscale, 0.25, anti_aliasing=False)
print(image_rescaled_25.shape)

#Task 4
ax = plt.hist(coins_grayscale.ravel(), bins = 256)

t = 120
binary = coins_grayscale < t


fig, ax = plt.subplots()
plt.imshow(binary, cmap="gray")
plt.show()

#Task 5
print("\nTask5")
image = data.coins() #¤Loads same coin image as above, but imported from skimage.data
coin = image[170:220, 75:130] #Selects a protion of the image with a coin in it. 
                            #The first two number selects in vertical direction and the second two in horizontal

result = match_template(image, coin) #Matches the pixel in the coin cropped image with the original sized image
ij = np.unravel_index(np.argmax(result), result.shape) #Gets coordinates of the match image
x, y = ij[::-1] #Puts coordintaes in x and y variables

fig = plt.figure(figsize=(8, 3)) #Creates 8x3 subplot figure, asically used as a 1x3 grid
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

ax1.imshow(coin, cmap=plt.cm.gray) # plots the single cion in "proper" grayscale color
ax1.set_axis_off()
ax1.set_title('template')

ax2.imshow(image, cmap=plt.cm.gray) #plots full image of coins
ax2.set_axis_off() #Turns off y- and x-axis
ax2.set_title('image') #Sets title
# highlight matched region
hcoin, wcoin = coin.shape #Gets the shape of coin
rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none') #Creates a red rectangle the same pixel size as coin
ax2.add_patch(rect) #adds the rectangle to the full coin image

ax3.imshow(result)  #plots the result of the matche_templade function
                    #Looks like a heatmap where the moore matched pixels have a green color and the non-mathched pixels
                    #has a blue color. Seems to be able to match the single coin to most of the other coins, even though they
                    #are a bit different. Seems like it matches on pixel "intensity", i.e. the grayscale value of the poxels.
ax3.set_axis_off()
ax3.set_title('`match_template`\nresult')
# highlight matched region
ax3.autoscale(False)
ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10) #Plots a ring at the original coin placement

plt.show()

