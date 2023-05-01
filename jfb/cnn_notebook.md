---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"id": "our_BnYUepoa"}

# Read data

Reading in the satellite data from a saved .npy file, and the classified data from a .tif. Plot the data. (.npy saved by save_sat_npy.py)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 378
id: KRy_tT9jvQRm
outputId: f3a73b90-feb2-44c4-a022-3a0313764c59
---
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import rasterio as rio
import earthpy.plot as ep
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.layers import Conv3D, Input, Reshape, Flatten, Conv2D, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import rioxarray
import a301_lib
import plotly.express as px
import pandas as pd
```

```{code-cell} ipython3
:id: dsnL3_L6G4Qo

# Reading in classified data.

def get_copern_xarray(lat, lon, size):
    lat_range = np.array([size, -size]) + lat
    lon_range = np.array([-size, size]) + lon

    copern_tif  = "./data/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif"
    copern_xarray = rioxarray.open_rasterio(copern_tif)
    copern_xarray = copern_xarray.sel(y=slice(lat_range[0], lat_range[1]), x=slice(lon_range[0], lon_range[1]))

    return copern_xarray

def get_feature_dict(band_xarray):
    flag_meanings = band_xarray.flag_meanings.split(", ")
    flag_values = list(map(int, band_xarray.flag_values.split(", ")))

    n_cats = len(flag_values)
    cat = np.arange(n_cats)

    val_to_cat = dict(zip(flag_values, cat))
    cat_to_mean = dict(zip(cat, flag_meanings))
    val_to_mean = dict(zip(flag_values, flag_meanings))
    return val_to_mean, cat_to_mean

# Converting data to have sequential values for plotting.
def convert_copern_xarray(copern_xarray):
    flag_meanings = copern_xarray.flag_meanings.split(", ")
    flag_values = list(map(int, copern_xarray.flag_values.split(", ")))

    n_cats = len(flag_values)
    cat = np.arange(n_cats)

    val_to_cat = dict(zip(flag_values, cat))
    cat_to_mean = dict(zip(cat, flag_meanings))

    out = copern_xarray.rio.transform()

    xarray_values = copern_xarray.data
    xarray_values = xarray_values.squeeze()

    #rewrite the classification ints
    vals = xarray_values
    vals_shape = xarray_values.shape
    vals = list(vals.flatten())

    sort_key = [val_to_cat[v] for v in vals]
    xarray_values = np.reshape(sort_key, vals_shape)
    
    return xarray_values
```

```{code-cell} ipython3
:id: zqAz56h1dYTt

# Call the above functions to get an array for ground truth and one with satellite data.

lat = 49.2827
lon = -123.120
size = 0.25
band_combinations = ["all", "false_colour"]
colors = band_combinations[0]
# jan, apr, july, oct
date_combinations = ["2019-01-01/2019-01-31", "2019-04-01/2019-04-30", "2019-07-01/2019-07-31", "2019-10-01/2019-10-31"]
date = date_combinations[2]

# Ground Truth
copern_xarray = get_copern_xarray(lat, lon, size)
val_to_mean, cat_to_mean = get_feature_dict(copern_xarray)
y_data = convert_copern_xarray(copern_xarray)
unique_classes = np.unique(y_data)

# Data
arr_st = np.load("./data/arr_st.npy")
data_shape = arr_st.shape[1:]
```

```{code-cell} ipython3
:id: AVebAAY0dbO_

# Plot satellite data
ep.plot_rgb(
    arr_st,
    rgb=(7, 3, 2),
)

plt.show()
```

```{code-cell} ipython3
:id: 5qeKV5sddbaj

# Plot classified data

fig = px.imshow(y_data)

fig.update_coloraxes(
    colorscale=px.colors.qualitative.Dark24,
    colorbar_tickvals=list(cat_to_mean.keys()),
    colorbar_ticktext=list(cat_to_mean.values()),
    colorbar_tickmode="array",
)

fig.show()
print(cat_to_mean) # to help reading the plot
```

+++ {"id": "lXqClkixerY-"}

# CNN
Setting up the CNN model. Following these links pretty closely.
- https://towardsdatascience.com/land-cover-classification-of-satellite-imagery-using-convolutional-neural-networks-91b5bb7fe808
- https://medium.com/geekculture/remote-sensing-deep-learning-for-land-cover-classification-of-satellite-imagery-using-python-6a7b4c4f570f
- https://github.com/syamkakarla98/Satellite_Imagery_Analysis

```{code-cell} ipython3
:id: KJL-f7AXd719

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels=False):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test
  
## GLOBAL VARIABLES
dataset = 'SB'
test_size = 0.30
windowSize = 15

X_data = np.moveaxis(arr_st, 0, -1)

# Apply PCA
K = 5
X,pca = applyPCA(X_data,numComponents=K)

print(f'Data After PCA: {X.shape}')

# Create 3D Patches
X, y = createImageCubes(X, y_data, windowSize=windowSize)
print(f'Patch size: {X.shape}')

# Split train and test
X_train, X_test, y_train, y_test = splitTrainTestSet(X, y, testRatio=test_size)

X_train = X_train.reshape(-1, windowSize, windowSize, K, 1)
X_test = X_test.reshape(-1, windowSize, windowSize, K, 1)

# One Hot Encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(f'Train: {X_train.shape}\nTest: {X_test.shape}\nTrain Labels: {y_train.shape}\nTest Labels: {y_test.shape}')
```

```{code-cell} ipython3
:id: NzYsdCvxd76u

S = windowSize
L = K
output_units = y_train.shape[1]

## input layer
input_layer = Input((S, S, L, 1))

## convolutional layers
conv_layer1 = Conv3D(filters=16, kernel_size=(2, 2, 3), activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=32, kernel_size=(2, 2, 3), activation='relu')(conv_layer1)
conv2d_shape = conv_layer2.shape
conv_layer3 = Reshape((conv2d_shape[1], conv2d_shape[2], conv2d_shape[3]*conv2d_shape[4]))(conv_layer2)
conv_layer4 = Conv2D(filters=64, kernel_size=(2,2), activation='relu')(conv_layer3)

flatten_layer = Flatten()(conv_layer4)

## fully connected layers
dense_layer1 = Dense(128, activation='relu')(flatten_layer)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(64, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
dense_layer3 = Dense(20, activation='relu')(dense_layer2)
dense_layer3 = Dropout(0.4)(dense_layer3)
output_layer = Dense(units=output_units, activation='softmax')(dense_layer3)
# define the model with input layer and output layer
model = Model(name = dataset+'_Model' , inputs=input_layer, outputs=output_layer)

model.summary()
```

+++ {"id": "Fkfeb_KLfEYO"}

# Training

```{code-cell} ipython3
:id: iIPDS5QJd7_o

# Compile
import os
from pathlib import Path

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# Callbacks
logdir = os.path.normpath("data/logs/" +model.name+'_'+datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))

tensorboard_callback = TensorBoard(log_dir=os.path.abspath(logdir))

es = EarlyStopping(monitor = 'val_loss',
                   min_delta = 0,
                   patience = 1,
                   verbose = 1,
                   restore_best_weights = True)

checkpoint = ModelCheckpoint(filepath = 'Pavia_University_Model.h5', 
                             monitor = 'val_loss', 
                             mode ='min', 
                             save_best_only = True,
                             verbose = 1)
# Fit
history = model.fit(x=X_train, y=y_train, 
                    batch_size=1024*6, epochs=6, 
                    validation_data=(X_test, y_test), callbacks = [tensorboard_callback, es, checkpoint])
```

```{code-cell} ipython3
:id: ncReeRE2fFc-

history = pd.DataFrame(history.history)

plt.figure(figsize = (12, 6))
plt.plot(range(len(history['accuracy'].values.tolist())), history['accuracy'].values.tolist(), label = 'Train_Accuracy')
plt.plot(range(len(history['loss'].values.tolist())), history['loss'].values.tolist(), label = 'Train_Loss')
plt.plot(range(len(history['val_accuracy'].values.tolist())), history['val_accuracy'].values.tolist(), label = 'Test_Accuracy')
plt.plot(range(len(history['val_loss'].values.tolist())), history['val_loss'].values.tolist(), label = 'Test_Loss')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.show()
```

# Results

```{code-cell} ipython3
:id: Wb2Ib-RnfFjw

pred = model.predict(X_test, batch_size=1204*6, verbose=1)

plt.figure(figsize = (10,7))

#classes = [f'Class-{i}' for i in range(1, n_classes+1)]
classes = [cat_to_mean[i] for i in unique_classes]

mat = confusion_matrix(np.argmax(y_test, 1),
                            np.argmax(pred, 1))

df_cm = pd.DataFrame(mat, index = classes, columns = classes)

sns.heatmap(df_cm, annot=True, fmt='d')

plt.show()
```

```{code-cell} ipython3
:id: MgqLtnIsfFnb

# Classification Report
print(classification_report(np.argmax(y_test, 1),
                            np.argmax(pred, 1),
      target_names = classes))
```

```{code-cell} ipython3
# Plotting the CNN classification results

pred_t = model.predict(X.reshape(-1, windowSize, windowSize, K, 1),
                       batch_size=1204*6, verbose=1)

fig = px.imshow(np.argmax(pred_t, axis=1).reshape(data_shape))

fig.update_coloraxes(
    colorscale=px.colors.qualitative.Dark24,
    colorbar_tickvals=list(cat_to_mean.keys()),
    colorbar_ticktext=list(cat_to_mean.values()),
    colorbar_tickmode="array",
)

fig.show()
print(cat_to_mean) # to help reading the plot
```

```{code-cell} ipython3

```
