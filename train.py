"""
Define model in Keras and train with TensorFlow Estimator API.
Create the Estimator using model_to_estimator().
"""
import os
import shutil
import datetime
import random
import argparse
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras import metrics
from camvid_dataset import dataset, random_crop
from keras_fc_densenet import build_FC_DenseNet56, build_FC_DenseNet67, build_FC_DenseNet103
from metrics import sparse_categorical_accuracy, mean_iou

tf.logging.set_verbosity(tf.logging.INFO)

# replace TF logging format
tf_logger = logging.getLogger('tensorflow')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
for handler in tf_logger.handlers:
    handler.formatter = formatter

def delete_dir(path):
    shutil.rmtree(path, ignore_errors=True)

def ts_rand():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    random_num = random.randint(1e6, 1e7-1)
    return '%s_%d' % (ts, random_num)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def distribution_strategy():
    num_gpus = len(get_available_gpus())
    tf.logging.info('Number of GPUs: %d' % num_gpus)
    if num_gpus == 0:
        tf.logging.info('Use OneDeviceStrategy with cpu:0')
        return tf.contrib.distribute.OneDeviceStrategy(device='/cpu:0')
    elif num_gpus == 1:
        tf.logging.info('Use OneDeviceStrategy with gpu:0')
        return tf.contrib.distribute.OneDeviceStrategy(device='/gpu:0')
    else:
        tf.logging.info('Use MirroredStrategy')
        return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)


    
def main(train_path, test_path, model_path, checkpoint_path, nb_epochs, batch_size, 
         image_height, image_width, crop_images, nb_crops):

    if checkpoint_path == None:
        model_dir = os.path.join(model_path, ts_rand())
        tf.logging.info('Create new checkpoint directory: %s' % model_path)
        os.makedirs(model_dir)
    else:
        tf.logging.info('Continue training from checkpoint directory: %s' % checkpoint_path)
        model_dir = checkpoint_path

    def train_input_fn():
        nb_cores = 16 # TODO configure dynamically
        ds = dataset(train_path, num_parallel_reads=nb_cores)
        if crop_images:
            tf.logging.info('Create %d crops %dx%d from each image' % (nb_crops, image_height, image_width))
            ds = ds.flat_map(random_crop(nb_crops=nb_crops, random_flip=True, 
                             crop_height=image_height, crop_width=image_width))
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=5000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=100)
        return ds

    def eval_input_fn():
        nb_cores = 16 # TODO configure dynamically
        ds = dataset(test_path, num_parallel_reads=nb_cores)
        if crop_images:
            # if the train images has been cropped we need to crop the eval 
            # images as well to get to the same input shape. The seed is 
            # fixed to create a stable eval dataset.
            ds = ds.flat_map(random_crop(nb_crops=1, random_flip=False, 
                             crop_height=image_height, crop_width=image_width,
                             seed=0))
        ds = ds.cache()
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=100)
        return ds

    fc_dn_model = build_FC_DenseNet103(nb_classes=32, final_softmax=True, input_shape=(image_height, image_width, 3), 
                                       data_format='channels_last')

    fc_dn_model.compile(optimizer=RMSprop(lr=1e-3),
                        loss='sparse_categorical_crossentropy',
                        metrics=[sparse_categorical_accuracy, mean_iou(num_classes=32)])

    accuracy = tf.get_default_graph().get_tensor_by_name('metrics/sparse_categorical_accuracy/Mean:0')
    tf.summary.scalar('accuracy', accuracy)


    run_config = tf.estimator.RunConfig(save_summary_steps=50, 
                                        keep_checkpoint_max=100



    # model_to_estimator creates a model_fn to initialize an Estimator. The 
    # EstimatorSpecs created by model_fn are initialized like this:
    #  - loss: will be set from model.total_loss
    #  - eval_metric_ops: will be set from model.metrics and model.metrics_tensors
    #  - predictions: will be set from model.output_names and model.outputs
    estimator = tf.keras.estimator.model_to_estimator(keras_model=fc_dn_model, 
                                                      model_dir=model_dir,
                                                      config=run_config)

    logged_tensors = {
        'global_step': tf.GraphKeys.GLOBAL_STEP, 
        'loss': fc_dn_model.total_loss.name,
        'accuracy': accuracy.name
        }

    best_loss = None
    best_loss_step = None
    for epoch in range(nb_epochs):
        tf.logging.info('Starting epoch %d' % epoch)
        train_hooks = [tf.train.LoggingTensorHook(tensors=logged_tensors, every_n_iter=1)]
        estimator.train(input_fn=train_input_fn, hooks=train_hooks)

        eval_hooks = [tf.train.LoggingTensorHook(tensors=logged_tensors, every_n_iter=1)]
        eval_results = estimator.evaluate(input_fn=eval_input_fn, hooks=eval_hooks)
        tf.logging.info('Epoch %d evaluation result: %s' % (epoch, eval_results))

        if best_loss == None or eval_results['loss'] < best_loss:
            best_loss = eval_results['loss']
            best_loss_step = eval_results['global_step']

        tf.logging.info('best_loss: %f at step: %d' % (best_loss, best_loss_step))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', help='train data path', default='./camvid-preprocessed/camvid-384x480-train.tfrecords')
    parser.add_argument('--test-path', help='test data path', default='./camvid-preprocessed/camvid-384x480-test.tfrecords')
    parser.add_argument('--model-path', help='base directory for checkpoints', default='./models')
    parser.add_argument('--checkpoint-path', help='directory with an existing checkpoint')
    parser.add_argument('--num-epochs', help='number of epochs', type=int, default=100)
    parser.add_argument('--batch-size', help='batch size', type=int, default=5)
    parser.add_argument('--image-height', help='model input image height', type=int, default=224)
    parser.add_argument('--image-width', help='model input image width', type=int, default=224)
    parser.add_argument('--crop-images', help='enable image cropping', action='store_true')
    parser.add_argument('--num-crops', help='number of crops per image, if cropping is enabled', type=int, default=5)
    args = parser.parse_args()
    main(train_path=args.train_path, test_path=args.train_path, model_path=args.model_path, 
         checkpoint_path=args.checkpoint_path, nb_epochs=args.num_epochs, batch_size=args.batch_size,  
         image_height=args.image_height, image_width=args.image_width, crop_images=args.crop_images, 
         nb_crops=args.num_crops)

