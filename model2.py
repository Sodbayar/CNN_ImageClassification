from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

#I used a smaller VGG16 model since I don't have an enough computing power(GPU)

size = 128
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(size, size, 3)))
model.add(layers.MaxPooling2D((3,3)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(3, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.0003), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
	rescale = 1./255,
	rotation_range=30,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.255)

#This is the labeling process
train_generator = train_datagen.flow_from_directory('data/train', target_size=(size,size), batch_size=64, classes=['candy', 'cattoy', 'sprite'])
validation_generator = validation_datagen.flow_from_directory('data/valid', target_size=(size,size), batch_size=64, classes=['candy', 'cattoy', 'sprite'])
#training
model.fit_generator(train_generator, epochs=5, steps_per_epoch=94, validation_data=validation_generator, validation_steps=10, workers=4)
model.save('model2.h5')