# See also the Jupyter notebook, exploration.ipynb, for a more detailed exploration.

# Imports
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# Put the image data (filenames and labels) into a pandas dataframe.
folders = ["with_mask","without_mask","mask_weared_incorrect"]
filenames = np.array([["./Dataset/"+folders[i]+"/"+str(j)+".png",folders[i]] for i in range(3) for j in range(3001)])
filenames = np.array([filenames[i] for i in range(len(filenames)) if os.path.exists(filenames[i][0])])
filenames = np.reshape(filenames,(filenames.shape[0],filenames.shape[1]))
img_df = pd.DataFrame(
    filenames,
    columns=['Filename', 'Label']
)

# Set up training, testing, and validation sets
# 80% test, 10% training, 10% validation
img_df = img_df.sample(frac=1).reset_index(drop=True)

train_size = int(len(img_df)*0.8)
test_size = int(( len(img_df) - train_size )/2)
validation_size = len(img_df) - train_size-test_size

train_df = img_df[:train_size]
test_df = img_df[train_size:train_size+test_size]
validation_df = img_df[train_size+test_size:]

# Load images
img_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_IMG_Set = img_generator.flow_from_dataframe(dataframe=train_df,
                                                       x_col="Filename",
                                                       y_col="Label",
                                                       color_mode="rgb",
                                                       class_mode="categorical",
                                                       target_size=(128,128),
                                                       subset="training")

validation_IMG_Set = img_generator.flow_from_dataframe(dataframe=validation_df,
                                                       x_col="Filename",
                                                       y_col="Label",
                                                       color_mode="rgb",
                                                       class_mode="categorical",
                                                       target_size=(128,128)
                                                    )

test_IMG_Set = img_generator.flow_from_dataframe(dataframe=test_df,
                                                       x_col="Filename",
                                                       y_col="Label",
                                                       color_mode="rgb",
                                                       class_mode="categorical",
                                                       target_size=(128,128),
                                                       shuffle=False)
                                                       
# The model
Model = tf.keras.models.Sequential()

Model.add(tf.keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=(128,128,3)))
Model.add(tf.keras.layers.BatchNormalization())
Model.add(tf.keras.layers.MaxPooling2D((2,2)))

Model.add(tf.keras.layers.Conv2D(64,(3,3),padding="same",activation="relu"))
Model.add(tf.keras.layers.Dropout(0.3))
Model.add(tf.keras.layers.MaxPooling2D((2,2)))

Model.add(tf.keras.layers.Conv2D(128,(3,3),padding="same",activation="relu"))
Model.add(tf.keras.layers.Dropout(0.3))
Model.add(tf.keras.layers.MaxPooling2D((2,2)))

Model.add(tf.keras.layers.Conv2D(128,(3,3),padding="same",activation="relu"))
Model.add(tf.keras.layers.Dropout(0.3))
Model.add(tf.keras.layers.MaxPooling2D((2,2)))

Model.add(tf.keras.layers.Flatten())
Model.add(tf.keras.layers.Dense(256,activation="relu"))
Model.add(tf.keras.layers.Dropout(0.5))

Model.add(tf.keras.layers.Dense(3,activation="softmax"))

Early_Stop = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=3,mode="min")

# Compile and train
Model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
CNN_Sep_Model = Model.fit(train_IMG_Set,validation_data=validation_IMG_Set,callbacks=Early_Stop,epochs=50)

Model_Results = Model.evaluate(test_IMG_Set)
print("LOSS:  " + "%.4f" % Model_Results[0])
print("ACCURACY:  " + "%.4f" % Model_Results[1])