import cv2
import numpy as np
from PIL import Image
from keras import models

#Load the saved model
model = models.load_model('model2.h5')
video = cv2.VideoCapture(0)

while True:
	_, frame = video.read()

	#Convert the captured frame into RGB
	im = Image.fromarray(frame, 'RGB')

	#Resizing into 128x128 because we trained the model with this image size
	im = im.resize((128, 128))
	img_array = np.array(im)

	#Our keras model used a 4D tensor, (imagexheightxwidthxchannel)
	img_array = np.expand_dims(img_array, axis=0)

	#prediction
	#one hot array to index
	prediction = np.argmax(model.predict(img_array)[0])
	print(prediction)
	if prediction == 0:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		print('Candy')
	elif prediction == 2:
		print('Sprite')
	else:
		print('Cattoy')

	cv2.imshow("Capturing", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
video.release()
cv2.destroyAllWindows()