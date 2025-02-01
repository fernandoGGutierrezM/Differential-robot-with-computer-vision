import cv2
import numpy as np

# Load YOLO model
#
net = cv2.dnn.readNet('final1.onnx')

# Load class names
#with open('coco.names', 'r') as f:
classes = ["tododerecho-pannel", "Stop-pannels" ,"Derecha-pannels", "Travaux-pannel"]

# Load image
image = cv2.imread('droite_im1.jpg')
height, width = image.shape[:2]

# Create blob from image
blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(224, 224), swapRB=True, crop=False)
net.setInput(blob)

# Run forward pass
layer_names = net.getLayerNames()
output_layers = [net.getLayer(i).name for i in net.getUnconnectedOutLayers()]
outs = net.forward(output_layers)

# Process detections
class_ids, confidences, boxes = [], [], []
for out in outs:
    for detection in out:
        #print(detection)
        scores = detection[1]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maxima suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#print(class_ids)
#print(classes)
#print(confidences[:])
print()
print(indices)

# Draw bounding boxes
'''
for i in indices:
    #i = i[0]
    box = boxes[i]
    x, y, w, h = box[0], box[1], box[2], box[3]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    text = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
'''
    
# Display image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
