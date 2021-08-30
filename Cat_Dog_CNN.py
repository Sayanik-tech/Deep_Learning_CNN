#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


tf.__version__


# **Data Preprocessing**

# Preprocessing The Training Data set

# In[4]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
        'A:/Udemy_ML_Training/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/Python/Section 40 - Convolutional Neural Networks (CNN)/dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# Preprocessing the Test Data Set

# In[5]:


test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'A:/Udemy_ML_Training/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/Python/Section 40 - Convolutional Neural Networks (CNN)/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# **Building CNN**

# Initializing CNN

# In[6]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
cnn = Sequential()


# 1.Convolution

# In[7]:


cnn.add(Conv2D(filters = 32 , kernel_size= 3 , activation = 'relu' , input_shape = [64,64,3]))


# 2.Pooling

# In[8]:


cnn.add(MaxPool2D(pool_size= 2, strides= 2))


# Adding a convolution Layer

# In[9]:


cnn.add(Conv2D(filters = 64 , kernel_size= 3 , activation = 'relu'))
cnn.add(MaxPool2D(pool_size= 2, strides= 2))


# 3.**Flattening**

# In[10]:


cnn.add(Flatten())


# 4.**Full Connection**

# In[11]:


cnn.add(Dense(128 , activation = 'relu'))


# 5**.Output Layer**

# In[12]:


cnn.add(Dense(1 , activation = 'sigmoid'))


# **Training The CNN**

# Compiling the CNN

# In[13]:


cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# Training teh CNN and Evaluating it on Test Set at the same time

# In[14]:


cnn.fit(x = training_set, validation_data= test_set,epochs = 25)                                                                                                                                      


# **Prediction**

# In[54]:


from keras.preprocessing import image
test_image = image.load_img('A:/Udemy_ML_Training\Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/Python/Section 40 - Convolutional Neural Networks (CNN)/dataset/single_prediction/cat_or_dog_999999999.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'Dog'
else:
  prediction = 'Cat' 


# In[55]:


print(prediction)


# In[ ]:





# In[ ]:




