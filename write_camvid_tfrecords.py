"""
Preprocess images and labels from the CamVid dataset (the 701 still images). 
The images are resized. The labels are mapped to unique int IDs and cropped. 

The script creates the following TFRecord files:
  * $OUTPUT_PATH/camvid-${IMAGE_HEIGHT}x${IMAGE_WIDTH}-train.tfrecords
  * $OUTPUT_PATH/camvid-${IMAGE_HEIGHT}x${IMAGE_WIDTH}-test.tfrecords
"""
import os
import glob
import argparse
from PIL import Image
import numpy as np
import tensorflow as tf
from camvid_utils import load_label_colors

def conv_label(label, label_code2id):
    """Map all color codes in `label_image` to code IDs.
       Args:
           label: array of shape [height, width, 3]
           label_code2id: dict that maps label codes to unique IDs
       Return:  
           array of shape [height, width, 1]
    """
    assert len(label.shape) == 3
    assert label.shape[2] == 3
    height = label.shape[0]
    width = label.shape[1]
    result = np.zeros((height, width), 'uint8')
    for j in range(height): 
        for k in range(width):
            try:
                result[j, k] = label_code2id[tuple(label[j, k])]
            except KeyError:
                print('Unknown label code: %s' % label[j, k])
                result[j, k] = label_code2id[tuple(np.array([0,0,0]))]

    return result

def open_image(image_path, image_size): 
    """Load an image and resize it.
       Args:
           image_path: path of image file
           image_size: target size
       Return:
           array of shape (height, width, 3)
    """
    return np.array(Image.open(image_path).resize(image_size, Image.NEAREST))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tf_records(tfrecords_path, images, labels):
    """Write images and labels as TFRecords to a file.
    Args:
    tfrecords_path: output path
    images: array with images
    labels: array with labels
    """
    with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
        for index in range(images.shape[0]):
            feature = { 
                        'height': int64_feature(images[index].shape[0]),
                        'width': int64_feature(images[index].shape[1]),
                        'label': bytes_feature(tf.compat.as_bytes(labels[index].tostring())), 
                        'image': bytes_feature(tf.compat.as_bytes(images[index].tostring()))
                      }
            example = tf.train.Example(features=tf.train.Features(feature=feature))    
            writer.write(example.SerializeToString())


def split_dataset(images, labels, test_fraction):
    nb_images = images.shape[0]
    idxs = np.arange(nb_images)
    np.random.shuffle(idxs)
    test_idxs = idxs[0:int(idxs.shape[0] * test_fraction)]
    train_idxs = idxs[int(idxs.shape[0] * test_fraction):]
    return images[train_idxs,:,:,:], labels[train_idxs,:], images[test_idxs,:,:,:], labels[test_idxs,:]

def main(input_path, output_path, image_height, image_width, test_fraction):

    images_path = os.path.join(input_path, '701_StillsRaw_full')
    labels_path = os.path.join(input_path, 'LabeledApproved_full')
    label_colors_path = os.path.join(input_path, 'label_colors.txt')

    train_out_path = os.path.join(output_path, 'camvid-%dx%d-train.tfrecords' % (image_height, image_width))
    test_out_path = os.path.join(output_path, 'camvid-%dx%d-test.tfrecords' % (image_height, image_width))

    os.makedirs(output_path, exist_ok=True) 

    image_size = (image_width, image_height)
    _, _, label_code2id = load_label_colors(label_colors_path)
    image_paths = glob.glob(os.path.join(images_path, '*.png'))

    print('Found %d classes' % len(label_code2id))

    num_images = len(image_paths)
    print('Found %d images' % num_images)

    # load all images in memory
    images = []
    labels = []
    for image_path in image_paths:
        print(image_path)
        image = open_image(image_path, image_size)
        images.append(image)
        label_path = os.path.join(labels_path, os.path.basename(image_path)[:-4] + '_L.png')
        label = open_image(label_path, image_size)
        label = conv_label(label, label_code2id)
        labels.append(label)
        
    images = np.array(images)
    labels = np.array(labels)

    labels = labels.reshape((labels.shape[0], labels.shape[1]*labels.shape[2]))
    train_images, train_labels, test_images, test_labels = split_dataset(images, labels, test_fraction)
    print('Split dataset into %d train and %d test images' % (train_images.shape[0], test_images.shape[0]))
    print('Train images: %s, %s' % (train_images.shape, train_images.dtype))
    print('Train labels: %s, %s' % (train_labels.shape, train_labels.dtype))
    print('Test images: %s, %s' % (test_images.shape, test_images.dtype))
    print('Test labels: %s, %s' % (test_labels.shape, test_labels.dtype))

    print('Writing:', train_out_path)
    write_tf_records(train_out_path, train_images, train_labels)
    print('Writing:', test_out_path)
    write_tf_records(test_out_path, test_images, test_labels)

    train_images = train_images / 255.
    print('Train image mean: %f' % np.mean(train_images))
    print('Train image std.: %f' % np.std(train_images))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', help='input directory', default='./camvid')
    parser.add_argument('--output-path', help='output directory', default='./camvid-preprocessed')
    parser.add_argument('--image-height', help='height for resizing images during loading', type=int, default=384)
    parser.add_argument('--image-width', help='width for resizing images during loading', type=int, default=480)
    parser.add_argument('--test-fraction', help='fraction of dataset to use for testing.', type=float, default=0.2)
    args = parser.parse_args()
    main(input_path=args.input_path, output_path=args.output_path, image_height=args.image_height, 
         image_width=args.image_width, test_fraction=args.test_fraction)
