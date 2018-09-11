# Keras Fully Connected DenseNet

This is a Keras implementation of the [Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/abs/1611.09326) paper.
The model is trained on the [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) dataset. 

## Train workflow

Clone Github repo with CamVid data

    git clone https://github.com/mostafaizz/camvid.git

Create TFRecord files

    python write_camvid_tfrecords.py --input-path ./camvid --output-path ./camvid-preprocessed --image-height 384 --image-width 480

Train model with cropped image size 224x224:

    python -u train.py \
        --train-path ./camvid-preprocessed/camvid-384x480-train.tfrecords \
        --test-path ./camvid-preprocessed/camvid-384x480-test.tfrecords \
        --model-path ./models \
        --image-height 224 \
        --image-width 224 \
        --batch-size 5 \
        --crop-images \
        --num-crops 5

Retrain model with full image size 384x480:

    python -u train.py \
        --train-path ./camvid-preprocessed/camvid-384x480-train.tfrecords \
        --test-path ./camvid-preprocessed/camvid-384x480-test.tfrecords \
        --checkpoint-path ${CHECKPOINT_PATH}
        --image-height 384 \
        --image-width 480 \
        --batch-size 5

The higher image size may cause OOM issues on some GPU devices. This can be solved by reducing the batch size.

## Classification examples

Here are the color encodings for the labels:

!["LabelsColorKey"](images/LabelsColorKey.jpg?raw=true "LabelsColorKey")

The following examples show the original image, the true label map and the predicted label map:

!["camvid-segmentation-1"](images/camvid-segmentation-1.png?raw=true "camvid-segmentation-1")

!["camvid-segmentation-1"](images/camvid-segmentation-2.png?raw=true "camvid-segmentation-2")

!["camvid-segmentation-3"](images/camvid-segmentation-3.png?raw=true "camvid-segmentation-3")

!["camvid-segmentation-4"](images/camvid-segmentation-4.png?raw=true "camvid-segmentation-4")

!["camvid-segmentation-5"](images/camvid-segmentation-5.png?raw=true "camvid-segmentation-5")

## Training metrics

Train metrics:

<img src="images/camvid_eval_loss.png" width="200" height="200"/>

!["camvid_eval_iou"](images/camvid_eval_iou.png)

<img src="images/camvid_eval_accuracy.png" width="200" height="200"/>


Retrain metrics:

!["camvid_eval_loss_retrain"](images/camvid_eval_loss_retrain.png?raw=true "camvid_eval_loss_retrain")

!["camvid_eval_iou_retrain"](images/camvid_eval_iou_retrain.png?raw=true "camvid_eval_iou_retrain")

!["camvid_eval_accuracy_retrain"](images/camvid_eval_accuracy_retrain.png?raw=true "camvid_eval_accuracy_retrain")
