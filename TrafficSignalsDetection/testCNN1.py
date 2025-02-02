import cv2
import numpy as np

class_names = ['tododerecho-pannel', 'Stop-pannels' ,'Derecha-pannels', 'Travaux-pannel']

COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

model = cv2.dnn.readNet('CNN_Models/final1.onnx')

image = cv2.imread('droite_im1.jpg')

image_height, image_width, _ = image.shape

# create blob from image
blob = cv2.dnn.blobFromImage(image=image, size=(224, 224), mean=(104, 117, 123), swapRB=True)
# set the blob to the model
model.setInput(blob)
# forward pass through the model to carry out the detection
outputs = model.forward()

final_outputs = outputs[0]
# make all the outputs 1D
final_outputs = final_outputs.reshape(1000, 1)
# get the class label
label_id = np.argmax(final_outputs)

print(label_id)
# convert the output scores to softmax probabilities
probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
# get the final highest probability
final_prob = np.max(probs) * 100.
# map the max confidence to the class label names
out_name = class_names[label_id]
out_text = f"{out_name}, {final_prob:.3f}"
# put the class name text on top of the image
cv2.putText(image, out_text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.imwrite('result_image.jpg', image)

