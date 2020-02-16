# 1. Importing Libraries


```python
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
```

# 2. Get Labels


```python
def getAllLabels(loc):
    dictLabel = {}
    
    for labels in os.listdir(loc):
        #print(labels)
        
        name = pd.read_fwf(loc+labels)
        new_name = name.iloc[0,0]
        labels = labels.replace('.txt','')
        #print(type(name))
        #break
        dictLabel[labels] = new_name
        
        #break
    return dictLabel
```


```python
allLabels = getAllLabels('../input/butterfly-dataset/leedsbutterfly/descriptions/')
```


```python
allLabels
```


```python
def getImages(loc):
    Labels = []
    Images = []
    
    for img in os.listdir(loc):
        
        image = cv2.imread(loc+img)
        image = cv2.resize(image,(120,120))
        
        Images.append(image)
        if img[:3] == '010':
            Labels.append('10')
        else:
            Labels.append(img[:3].replace('0',""))
        
    return shuffle(Images,Labels)
```


```python
Images,Labels = getImages('../input/butterfly-dataset/leedsbutterfly/images/')
```


```python
Images = np.array(Images)
Labels = np.array(Labels)
```


```python
Images.shape
```


```python
Labels.shape
```


```python
Images = Images/255
```


```python
Images.max()
```


```python
Images.min()
```


```python
Images.dtype
```


```python
Labels = Labels.astype(int)
```


```python
Labels.dtype
```


```python
type(Labels)
np.unique(Labels)
```


```python
import keras
Labels = keras.utils.to_categorical(Labels,num_classes=11)
```


```python
x_train,x_test,y_train,y_test = train_test_split(Images,Labels,test_size=.2)
```


```python
x_train.shape
```


```python
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_datagen.fit(x_train)
```


```python
from keras.applications.vgg16 import VGG16
base_model = VGG16(weights='imagenet',include_top=False,input_shape=(120,120,3))
```


```python
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,MaxPool2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
```


```python
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1024,activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(11,activation='softmax')(x)

model = Model(inputs=base_model.input,outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
```


```python
model.summary()
```


```python
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```


```python
historyd = model.fit_generator(train_datagen.flow(x_train,y_train,batch_size=32),
                               validation_data=(x_test,y_test),epochs=30)
```


```python
model.save('my_model.h5') 

from IPython.display import FileLink
FileLink('my_model.h5')
```


```python

```
