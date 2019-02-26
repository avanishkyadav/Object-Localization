from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten,Dropout, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.optimizers import Adam

def SkipNet(X, f, filters):
    F1, F2, F3 = filters
    Xh = X
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = Conv2D(F2,(f,f),strides=(1,1),padding='same',kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(F3,(1,1),strides=(1,1),padding='valid',kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3)(X)
    X = Add()([X,Xh])
    X = Activation('relu')(X)
    return X

def ConvNet(X, f, filters, s = 2):
    F1, F2, F3 = filters
    Xh = X
    X = Conv2D(F1, (1, 1), strides = (s,s), kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = Conv2D(F2,(f,f),strides=(1,1),padding='same',kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(F3,(1,1),strides=(1,1),padding='valid',kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3)(X)
    Xh = Conv2D(F3,(1,1),strides=(s,s),padding='valid',kernel_initializer=glorot_uniform())(Xh)
    Xh = BatchNormalization(axis=3)(Xh)
    X = Add()([X,Xh])
    X = Activation('relu')(X)
    return X

def ResNet(input_shape = (90, 120, 3)):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3,)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = ConvNet(X, f = 3, filters = [64, 64, 256], s = 1)
    X = SkipNet(X, 3, [64, 64, 256])
    X = SkipNet(X, 3, [64, 64, 256])

    X = ConvNet(X,3,[128,128,512])
    X = SkipNet(X,3,[128,128,512])
    X = SkipNet(X,3,[128,128,512])
    X = SkipNet(X,3,[128,128,512])

    X = ConvNet(X,3,[256,256,1024],s=2)
    X = SkipNet(X,3,[256,256,1024])
    X = SkipNet(X,3,[256,256,1024])
    X = SkipNet(X,3,[256,256,1024])
    X = SkipNet(X,3,[256,256,1024])
    X = SkipNet(X,3,[256,256,1024])

    X = ConvNet(X,3,[512,512,2048],s=2)
    X = SkipNet(X,3,[512,512,2048])
    X = SkipNet(X,3,[512,512,2048])

    X = ConvNet(X,3,[1024,1024,4096],s=2)
    X = SkipNet(X,3,[1024,1024,4096])

    X = AveragePooling2D((2,2))(X)
    X = Flatten()(X)
    X = Dense(1024, kernel_initializer = glorot_uniform())(X)
    out = Dense(4, kernel_initializer = glorot_uniform())(X)

    model = Model(inputs = X_input, outputs = out, name='ResNet')

    return model
