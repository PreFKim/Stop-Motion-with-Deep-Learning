import keras 

def conv(input,features,kernel_size=3,strides = 1,padding='same',is_relu=True,is_bn=False):
    x= keras.layers.Conv2D(features,kernel_size,strides,padding,kernel_initializer='he_normal')(input)
    if is_bn:
        x = keras.layers.BatchNormalization()(x)
    if is_relu:
        x = keras.activations.relu(x)
    return x