from keras.layers import *
from keras.models import *

def sub_network(input_shape, dropout=0.5):
    c = [32, 64, 64]
    
    inputs = Input(shape=input_shape)
    
    x = Conv2D(c[0], (3, 3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(dropout)(x)

    x = Conv2D(c[1], (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(dropout)(x)

    x = Conv2D(c[2], (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(dropout)(x)
    
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(128)(x)

    model = Model(inputs=inputs, outputs=x)
    
    model.summary()

    return model