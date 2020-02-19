import numpy as np
import cv2
from scipy.spatial import distance as distance
from collections import OrderedDict
import CentroidTracker

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# Open webcam video stream 0 for built-in webcam, 1 for external webcam
cap = cv2.VideoCapture(2)

# The output will be written to webcam_output.avi
out = cv2.VideoWriter("webcam_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15., (640, 480))
ct = CentroidTracker.CentroidTracker()
(H, W) = (None, None)

while(True):
    # Reading the frame
    ret, frame = cap.read()

    # Resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    # Using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if W is None or H is None:
        (H, W) = frame.shape[:2]
	# construct a blob from the frame, pass it through the network,
	# obtain our output predictions, and initialize the list of
	# bounding box rectangles
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0))
    #net.setInput(blob)
    #detections = net.forward()
    rects = []

    # Detect people in the image, returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))

    boxes = np.array([[x, y, x+w, y+h] for (x, y, w, h) in boxes])
    print("running...")

	# if the frame dimensions are None, grab them
    
    for i in boxes:

        rects.append(i)
        cv2.rectangle(frame, i, (0, 255, 0), 2)

    objects = ct.update(rects)
	# loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
		# object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


    # Write the output video
    out.write(frame.astype("uint8"))
    # Display the resulting frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything is done, release the capture
cap.release()
# And release the output
out.release()
# Finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
