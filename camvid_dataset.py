"""
tf.Dataset for the CamVid dataset.
"""
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from camvid_utils import normalize_image

def _parse_tf_record(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
        })

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [height, width, 3])
    image = tf.cast(image, tf.float32)
    label = tf.decode_raw(features['label'], tf.uint8)
    label = tf.reshape(label, [height * width])
    label = tf.cast(label, tf.int32)
    image = normalize_image(image)
    return image, label

def _with_dependencies(deps, tensor):
    """
    Args:
        deps: A list of Operations or Tensors which must be executed 
            or computed before accessing `tensor`. 
        tensor: The tensor to define the dependencies on    
    Return:
        A Tensor. Has the same type as `tensor`.
    """
    with tf.control_dependencies(deps):
        tensor = tf.identity(tensor)
    return tensor


def random_crop(nb_crops, random_flip=False, crop_height=224, crop_width=224, seed=None):
    def random_crop_fn(image, label):
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        assert_crop_height = tf.Assert(tf.greater_equal(image_height, crop_height), 
                                       [image_height], name='assert_image_height_ge_crop_height')
        image_height = _with_dependencies([assert_crop_height], image_height)

        assert_crop_width = tf.Assert(tf.greater_equal(image_width, crop_width), 
                                       [image_width], name='assert_image_width_ge_crop_width')
        image_width = _with_dependencies([assert_crop_width], image_width)

        # concatenate image and label so we can crop them in one operation
        label = tf.reshape(label, [image_height, image_width, 1])
        block = tf.concat([image, tf.cast(label, tf.float32)], axis=-1)
        
        ds = None
        for _ in range(nb_crops):
            crop = tf.random_crop(block, [crop_height, crop_width, 4], seed=seed)
            if random_flip:
                crop = tf.image.random_flip_left_right(crop, seed=seed)

            image_crop = crop[:,:,:3]
            label_crop = tf.cast(crop[:,:,-1], tf.int32)
                                        
            label_crop = tf.reshape(label_crop, [crop_height * crop_width])
            if ds == None:
                ds = tf.data.Dataset.from_tensors((image_crop, label_crop))
            else:
                ds = ds.concatenate(tf.data.Dataset.from_tensors((image_crop, label_crop)))
        return ds        
    return random_crop_fn

def dataset(tf_record_paths,
            compression_type=None,
            buffer_size=None,
            num_parallel_reads=None):
    """Returns a tf.Dataset with CamVid images and labels.
    Args:
       tf_record_paths: string or list of strings that are path(s) to TFRecord
           files.
       compression_type: A string evaluating to one of "" (no compression), "ZLIB", 
           or "GZIP".
       buffer_size: A int64 scalar representing the number of bytes in the read buffer. 
           0 means no buffering.
       num_parallel_reads: A int64 scalar representing the number of files to read 
           in parallel. Defaults to reading files sequentially.
    """
    ds = tf.data.TFRecordDataset(tf_record_paths, compression_type, 
                                 buffer_size, num_parallel_reads)
    ds = ds.map(_parse_tf_record, num_parallel_calls=num_parallel_reads)
    return ds

def load_camvid_tfrecords(input_path, max_images=None):
    """Load the CamVid dataset and return as Numpy arrays.
       The images will be normalized.
       Args:
           input_path: path of a TFRecord file
           max_images: max. number of images to return
       Return:
           images: float32 array with shape (None, height, width, 3)
           labels: int32 array with shape (None, height * width)
    """
    ds = dataset(input_path) # camvid_dataset.dataset() returns normalized images!
    iterator = ds.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        images = []
        labels = []
        while True:
            try:
                image, label = sess.run(next_element)
                images.append(image)
                labels.append(label)
                if len(images) == max_images:
                  break
            except tf.errors.OutOfRangeError:
                break
                
        images = np.array(images)
        labels = np.array(labels)
        return images, labels

