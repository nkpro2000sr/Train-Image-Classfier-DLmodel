import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from tensorflow.keras.layers import Conv2D,MaxPooling2D

import os, logging, json

def train(tdd, vdd, tmf, shape, epochs = 25, learning_rate = 0.001,
          earlystop_monitor = 'val_loss', # False for disable earlystop
          n_cnn_layers = 4, batch_size = 32, show_graph = True) :
    """
    single function for traning model for all types of image classification
    eg: emotion detection, gender classification, etc.,

    $tdd = path of the traning dataset directory
    $vdd = path of the validation dataset directory
    > dataset directory have subdirs with dirname= label
    > and each subdir have pics of their respective lable

    $tmf = path to save the trained model
    $shape = pixel size of the image to be reshaped
    """
    
    jd = dict()

    train_data_dir = tdd; jd["train_data_dir"]= train_data_dir
    validation_data_dir = vdd; jd["validation_data_dir"]= validation_data_dir
    trained_model_file = tmf; jd["trained_model_file"]= trained_model_file

    jd["labels"]= next(os.walk(train_data_dir))[1]
    assert (next(os.walk(train_data_dir))[1] == next(os.walk(validation_data_dir))[1]),\
           "labels must be same in both traning dataset and validation dataset"
    num_classes = len(next(os.walk(train_data_dir))[1]) # no. of subdirectories in train_data_dir
    jd["n_labels"]= num_classes
    image_shape = shape # (r,c,p) => Image with `r` x `c` pixcels and `p` values in each pixcel
    #                          p = 1 for grayscape and 3 for rgb or 4 for rgpa
    jd["image_shape"]= shape

    # generating multiple images with different aspects from our train_data

    if image_shape[2] == 1:
        mode = 'grayscale'
    elif image_shape[2] == 3:
        mode = 'rgb'
    else :
        mode = 'rgba'

    train_datagen = ImageDataGenerator(
                        rescale=1./255,
                        rotation_range=30,
                        shear_range=0.3,
                        zoom_range=0.3,
                        width_shift_range=0.4,
                        height_shift_range=0.4,
                        horizontal_flip=True,
                        fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(
                        train_data_dir,
                        color_mode=mode,
                        target_size=(image_shape[0],image_shape[1]),
                        batch_size=batch_size,
                        class_mode='categorical',
                        shuffle=True)

    # generating multiple images with different aspects from our validation_data

    validation_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = validation_datagen.flow_from_directory(
                                validation_data_dir,
                                color_mode=mode,
                                target_size=(image_shape[0],image_shape[0]),
                                batch_size=batch_size,
                                class_mode='categorical',
                                shuffle=True)

    # creating model

    model = Sequential()

    ## Convolutional Neural Network (CNN) Layers

    get_filters_int = lambda n : batch_size*(2**(n+1)) # to get filter in each layer (iteration)

    model.add(Conv2D(batch_size,(3,3),padding='same',kernel_initializer='he_normal',input_shape= image_shape ))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(batch_size,(3,3),padding='same',kernel_initializer='he_normal',input_shape= image_shape ))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    for i in range(number_of_cnn_layers-1):
        model.add(Conv2D(get_filters_int(i),(3,3),padding='same',kernel_initializer='he_normal'))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(get_filters_int(i),(3,3),padding='same',kernel_initializer='he_normal'))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

    ## Fully Connected Layer

    model.add(Flatten()) # flattened matrix into vector

    model.add(Dense(64,kernel_initializer='he_normal'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(64,kernel_initializer='he_normal'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(num_classes,kernel_initializer='he_normal'))
    model.add(Activation('softmax'))

    model_summary = model.summary()
    logging.info(model_summary) # summary of model
    jd["model_summary"]= model_summary

    ## for monitor the training ( to stop training if learing_accuracy_curve falls )

    from tensorflow.keras.optimizers import RMSprop,SGD,Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

    if earlystop_monitor :

        checkpoint = ModelCheckpoint(trained_model_file, # saving model_file
                                monitor = earlystop_monitor,
                                mode='min',
                                save_best_only=True,
                                verbose=1)

        earlystop = EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=3,
                            verbose=1,
                            restore_best_weights=True
                            )

        reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                factor=0.2,
                                patience=3,
                                verbose=1,
                                min_delta=0.0001)

        callbacks = [earlystop,checkpoint,reduce_learning_rate]

    model.compile(loss='categorical_crossentropy',
                optimizer = Adam(lr=learning_rate),
                metrics=['accuracy']) # we need accuracy

    ## training model

    number_of_train_samples = len([j for _,_,i in os.walk(train_data_dir) for j in i])
    number_of_validation_samples = len([j for _,_,i in os.walk(validation_data_dir) for j in i])

    if earlystop_monitor :

        history=model.fit_generator(
                    train_generator,
                    steps_per_epoch=number_of_train_samples//batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=validation_generator,
                    validation_steps=number_of_validation_samples)

    else :

        history=model.fit_generator(
                    train_generator,
                    steps_per_epoch=number_of_train_samples//batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=number_of_validation_samples)

    model.save(trained_model_file)

    #model.save_best_only(''.join(trained_model_file.rsplit('.',1).insert(-1,'Best')))

    history = history.history
    jd["history"]= history

    with open(trained_model_file+".json",'w') as jd_file :
        json.dump(jd, jd_file)

    import matplotlib.pyplot as plt
    plt.style.use("ggplot")

    y1=history["loss"]
    y2=history["val_loss"]
    plt.plot(y1,label="loss")
    plt.plot(y2,label="val_loss")
    plt.legend()

    plt.style.use("ggplot")

    y1=history["accuracy"]
    y2=history["val_accuracy"]
    plt.plot(y1,label="accuracy")
    plt.plot(y2,label="val_accuracy")
    plt.legend()
    if show_graph : plt.show()

    return jd

if __name__ == "__main__" :
    train(
        "dataset_for_training",
        "dataset_for_validation",
        "model.h5",
        (48,48,1)
        )
