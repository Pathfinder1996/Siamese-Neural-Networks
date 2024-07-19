from keras.layers import *
from keras.models import *

def sub_network(input_shape):

    inputs = Input(input_shape)

    x = Conv2D(32, (3, 3), activation="relu")(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(128)(x)

    model = Model(inputs=inputs, outputs=x)
    
    model.summary()

    return model