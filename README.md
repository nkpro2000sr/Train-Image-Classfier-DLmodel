# Train-Image-Classfier-DLmodel
To train IMAGE CLASSFIER Deep Learning model using CNN. Easy_To_use

### structure of dataset
```ascii_diagrame
                      +-----------------+
                      | Training Dataset|
                      +--------+--------+
                               |
                               v
                +------+-------+-------------+
                v      v                     v
          +-----+--+ +-+-------+      +------+--+
          |label_1 | | label_2 |......|label_n  |
          +--+-----+ +-----+---+      +-----+---+
             |             |                |
             v             v                v
     +---+---+-----+  ......................
     v   v         v
+----+-+ +------+  +------+
|img1_1| |img1_2|..|img1_n|   ...............
+------+ +------+  +------+
# same for validation dataset
```
> for emotion detection (label_1,label_2,...) = ("happy","sad",...) and
> (img1_1,img1_2,...) are images of happy faces, (img2_1,img2_2,...) are images of sad faces, ...

> similarly for gender detection (label_1,label_2) = ("Male","Female") and
> (img1_1,img1_2,...) are images of Male & (img2_1,img2_2,...) are images of Female.

#### model.json
this have detail about the model
* training_data_dir : path of training dataset
* validation_data_dir : path of validation dataset
* trained_model_file : path of trained model
* labels : list of labels (output)
* n_labels : no. of labeels
* image_shape : shape of image used for training and prediction
* dataset : full tree of dataset (including both training dataset and validation dataset)
* image_mode : mode of image (grayscale|rgb|rgba)
* model_summary : summary about the model
* number_of_training_samples : no. of training samples (no. of images used for training)
* number_of_validation_samples : no. of validation samples (no. of images used for validation)
* history : history.history gentated while model training (model.fit_generator -> history)

helpful at prediction
```python3
with open("model.json",'r') as file:
    model_dict = json.load(file)
predicted_label = model_dict["labels"][np.argmax(output_vector_from_model)]
```
