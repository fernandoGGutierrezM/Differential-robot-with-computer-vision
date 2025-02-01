import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt

def plot_histogram(image, title, mask=None):
	
	# split the image into its respective channels, then initialize
	# the tuple of channel names along with our figure for plotting
    chans = cv2.split(image)
    colors = ("r", "g", "b")
    #plt.figure()
    #plt.title(title)
    #plt.xlabel("Bins")
    #plt.ylabel("# of Pixels")
    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and plot it
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
        
    plt.show()
    
    # Estos son para plotear en tiempo real el histograma
    #plt.pause(0.01)
    #plt.clf()
    return 0

def main():
    #cap = cv2.VideoCapture(0)
    frame = cv2.imread('imgSemaphore_Yellow.png')
    
    color = np.copy(frame)
    arreglo = np.array(color)

    while True:
        #ret, frame = cap.read()
        #if not ret:
        #    break
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        alow = gray.min()
        ahigh = gray.max()
        print("MIN: ", alow)
        print("MAX: ", ahigh)
        amax = 89
        amin = 0
        (B, G, R) = cv2.split(frame)

        alpha = ((amax-amin)/(ahigh-alow))
        beta = amin - alow * alpha
        #print("alpha: ", frame.mean())
        #alpha = ((ahigh-alow/(amax-amin)))
        
        brightImage = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
        plot_histogram(brightImage, "Histograma")
        
        
        cv2.imshow('imagen: ', frame)
        cv2.imshow('brillo', brightImage)



#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break    
    #cap.release()
    cv2.destroyAllWindows()

    
	
if __name__ == "__main__":
	main()
#THIS IS THE LAST ONE
# Green:  72 47 56 | 115 77 68
# Red:    67 60 71 | 96 91 83
# Yellow: 60 54 57 | 96 88 72

# Para verde:    158 177 110 | 178 198 150
# Para Rojo:     185 55 33   | 248 94 71
# Para amarillo: 165 149 80  | 211 190 112
'''
#video_path = 'mp4TestLaneDetection4.mp4'
image_path = 'imgTestLaneDetection1.png'
#cap  = cv2.VideoCapture(video_path)


def plot_histogram(image, title, mask=None):
	# split the image into its respective channels, then initialize
	# the tuple of channel names along with our figure for plotting
	chans = cv2.split(image)
	colors = ("b", "g", "r")
	plt.figure()
	plt.title(title)
	plt.xlabel("Bins")
	plt.ylabel("# of Pixels")
	# loop over the image channels
	for (chan, color) in zip(chans, colors):
		# create a histogram for the current channel and plot it
		hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
		plt.plot(hist, color=color)
		plt.xlim([0, 256])

image = cv2.imread(image_path)
plot_histogram(image, "SADASDSA")
plt.show()
cv2.imshow(image)

'''