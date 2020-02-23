import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from tensorflow.keras.layers import Conv2D,MaxPooling2D

import os, logging, json

def train(p_Tds, p_Vds, p_saveModel, shape,
          epochs = 25, learning_rate = 0.001, #trade-off between time and accuracy
          callbacks = 'early_stop_val_loss_min',
          IDG_kwargs = -1,
          n_cnn_layers = 4, batch_size = 32,
          get_filters = -2, get_kernel_size = -2, get_pool_size = -1,
          save_json = True, show_graph = True) :
    """
    single function for traning model for all types of image classification
    eg: emotion detection, gender classification, etc.,

    $p_Tds = path of the training dataset directory
    $p_Vds = path of the validation dataset directory
     dataset directory have subdirs with dirname= label
     and each subdir have pics of their respective lable

    $p_saveModel = path to save the trained model
    $shape = pixel size of the image to be reshaped

    $callbacks = list => list of callables called while training model
     = 'early_stop_val_loss_min' => use early_stop_val_loss_min callbacks (see source code)
     = False => for no callbacks

    $get_filters is a function to get filters for each Conv2D layers
     = lambda i:filters_list[i-1] #i is index of each layers
     = -2 => use recommended lambda i : batch_size*(2**(i-1)) #similar_to VGG16 architecture
    $get_kernel_size is a function to get kernel_size for each Conv2D layers
     = lambda i:kernel_size_list[i-1]
     = -2 => use recommended lambda i : (3,3)
    $get_pool_size is a function to get pool_size for each Con2D layers
     = lambda i:pool_size_list[i-1]
     = -1 => use default
     = -2 => use recommended lambda i : (2,2)

    $IDG_kwargs is a dict of {kew_word_arguments:values} in ImageDataGenerator
     = dict => which is passed as keyword arguments
     = -1 => use default (same as `IDG_kwargs=dict()`)
     = -2 => use recommended

    $save_json = True to save the jd (info_dict) as p_saveModel+'.json'
     = path to save json file
     = False to don't
    $show_graph = True to show graph of ['loss','val_loss','accuracy','val_accuracy'] at training 
     = False to don't
    """
    
    jd = dict()

    training_data_dir = p_Tds; jd["training_data_dir"]= training_data_dir
    validation_data_dir = p_Vds; jd["validation_data_dir"]= validation_data_dir
    trained_model_file = p_saveModel; jd["trained_model_file"]= trained_model_file

    jd["labels"]= next(os.walk(training_data_dir))[1]
    assert (jd["labels"] == next(os.walk(validation_data_dir))[1]),\
           "labels must be same in both traning dataset and validation dataset"
    num_classes = len(jd["labels"]) # no. of subdirectories in training_data_dir
    jd["n_labels"]= num_classes
    image_shape = shape # (r,c,p) => Image with `r` x `c` pixcels and `p` values in each pixcel
    #                          p = 1 for grayscape and 3 for rgb or 4 for rgpa
    jd["image_shape"]= shape

    jd["dataset"] = dict(training= defaultdict(list), validation= defaultdict(list))
    for label in jd["labels"]:
        for file in os.listdir(os.path.join(training_data_dir,label)):
            jd["dataset"]["training"][label].append(os.path.join(training_data_dir,label,file))
        for file in os.listdir(os.path.join(validation_data_dir,label)):
            jd["dataset"]["validation"][label].append(os.path.join(validation_data_dir,label,file))

    # generating multiple images with different aspects from our train_data

    if image_shape[2] == 1:
        mode = 'grayscale'
    elif image_shape[2] == 3:
        mode = 'rgb'
    else :
        mode = 'rgba'

    if IDG_kwargs == -1 : IDG_kwargs = dict()
    elif IDG_kwargs == -2 :
        IDG_kwargs = dict(
                        rescale=1./255,
                        rotation_range=30,
                        shear_range=0.3,
                        zoom_range=0.3,
                        width_shift_range=0.4,
                        height_shift_range=0.4,
                        horizontal_flip=True,
                        fill_mode='nearest')

    train_datagen = ImageDataGenerator(rescale=1./255, **IDG_kwargs)

    train_generator = train_datagen.flow_from_directory(
                        training_data_dir,
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
                                target_size=(image_shape[0],image_shape[1]),
                                batch_size=batch_size,
                                class_mode='categorical',
                                shuffle=True)

    # creating model

    model = Sequential()

    ## Convolutional Neural Network (CNN) Layers

    if get_filters == -2 : get_filters = lambda i : batch_size*(2**(i-1)) # to get filter in each layer (iteration)
    if get_kernel_size == -2 : get_kernel_size = lambda i : (3,3) # to get kernel_size in each layer (iteration)
    if get_pool_size == -2 : get_pool_size = lambda i : (2,2) # to get pool_size in each MaxPooling2D layer

    # first layer i == 1
    model.add(Conv2D(get_filters(1),get_kernel_size(1),padding='same',kernel_initializer='he_normal',input_shape= image_shape ))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(get_filters(1),get_kernel_size(1),padding='same',kernel_initializer='he_normal',input_shape= image_shape ))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    if get_pool_size != -1 : model.add(MaxPooling2D(pool_size=get_pool_size(1)))
    else : model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    for i in range(number_of_cnn_layers-1):
        model.add(Conv2D(get_filters(i+2),get_kernel_size(i+2),padding='same',kernel_initializer='he_normal'))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(get_filters(i+2),get_kernel_size(i+2),padding='same',kernel_initializer='he_normal'))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        if get_pool_size != -1 : model.add(MaxPooling2D(pool_size=get_pool_size(i+2)))
        else : model.add(MaxPooling2D())
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

    fg_kwargs = dict()

    if callbacks == 'early_stop_val_loss_min' :

        checkpoint = ModelCheckpoint(trained_model_file, # saving model_file
                                save_best_only=True,
                                monitor='val_loss',
                                mode='min',
                                verbose=1)

        earlystop = EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=3,
                            verbose=1,
                            restore_best_weights=True)

        reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                factor=0.2,
                                patience=3,
                                verbose=1,
                                min_delta=0.0001)

        fg_kwargs['callbacks'] = [earlystop,checkpoint,reduce_learning_rate]

    #TODO add more defined callback methods

    elif callbacks :

        fg_kwargs['callbacks'] = callbacks

    model.compile(loss='categorical_crossentropy',
                optimizer = Adam(lr=learning_rate),
                metrics=['accuracy']) # we need accuracy

    ## training model

    number_of_train_samples = len([j for _,_,i in os.walk(training_data_dir) for j in i])
    jd["number_of_train_samples"] = number_of_train_samples
    number_of_validation_samples = len([j for _,_,i in os.walk(validation_data_dir) for j in i])
    jd["number_of_validation_samples"] = number_of_validation_samples

    #TODO add checkpoints and auto restore them to resume training anywhere and anytime
    history=model.fit_generator(
                    train_generator,
                    steps_per_epoch=number_of_train_samples//batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=number_of_validation_samples,
                    **fg_kwargs)

    model.save(trained_model_file)

    #model.save_best_only(''.join(trained_model_file.rsplit('.',1).insert(-1,'Best')))

    history = history.history
    jd["history"]= history

    if save_json :
        with open(save_json if type(save_json)==str else trained_model_file+".json",'w') as jd_file :
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
