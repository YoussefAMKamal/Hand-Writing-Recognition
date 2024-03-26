#%%
import os
import numpy
from PIL import Image
import keras
import matplotlib.pyplot as plt
import warnings

import buildModel

warnings.simplefilter("ignore")

file = "model.keras"
if os.path.isfile(file) == False:
    buildModel.fit_model()

model = keras.models.load_model(file)

loc = "Samples\\"
no = 0

for imgName in os.listdir(loc):
    no += 1
    print("----------------------------------------------------")
    print("Test Number {}:\nImage used for this test:".format(no))
    Img_path = os.path.join(loc, imgName)
    plt.figure(figsize=(2, 2))
    img = Image.open(Img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    img = img.resize((28,28)).convert('L')
    print("Predicting...")
    img = numpy.expand_dims(img, axis=0)
    prediction = numpy.argmax(model.predict(img))
    print("Prediction result: The image belong to class {}".format(prediction))
# %%