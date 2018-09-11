import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.training.saver import latest_checkpoint
from tensorflow.keras.optimizers import RMSprop
from keras_fc_densenet import build_FC_DenseNet103
from camvid_utils import load_label_colors, color_label, normalize_image

def main():

    video_path = './01TP_extract.avi'
    label_colors_path = './label_colors.txt'
    model_dir = '/tmp/retrained_model'

    label_colors, _, _ = load_label_colors(label_colors_path)

    checkpoint_path = latest_checkpoint(model_dir)
    tf.logging.info('Latest checkpoint:', checkpoint_path)

    model = build_FC_DenseNet103(nb_classes=32, final_softmax=True, 
                             input_shape=(224, 224, 3))
    model.compile(optimizer=RMSprop(lr=1e-4), 
    	          loss='sparse_categorical_crossentropy')

    # Loading the checkpoint without an Estimator is currently a hack.
    # The problem is that the model mixes Keras layers with Tensorflow scopes.
    # After compiling the model, all variables have full qualified names (with
    # correct scopes). But the checkpoint only contains the original Keras
    # variable names. The quick fix is to map the variable names. The proper 
    # solution could be to remove the TensorFlow scopes in the model and maybe 
    # use fully qualified names for the Keras layers.
    var_list = {}
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        full_name = v.name[:-2]
        parts = full_name.split('/')
        keras_name = '/'.join(parts[-2:])
        var_list[keras_name] = v

    saver = tf.train.Saver(var_list=var_list)

    tf.logging.info('Open video')
    cap = cv2.VideoCapture(video_path)
    
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    # Define the fps to be equal to 10. Also frame size is passed.
    frame_width = 224
    frame_height = 224
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    with tf.Session() as sess:

        saver.restore(sess, checkpoint_path)

        nb_frames = 0
        while(cap.isOpened()):

            ret_val, frame = cap.read()
            if not ret_val:
                break
            nb_frames += 1    
            image = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)

            image = normalize_image(image)

            images = np.expand_dims(image, axis=0)

            probs = sess.run(model.output, {model.input: images}) # probs.shape: (batch_size, 224*224, 32)
            predicted_labels = np.argmax(probs, axis=-1) # predicted_labels.shape: (batch_size, 224*224)
            y_pred = predicted_labels[0].reshape((images.shape[1],images.shape[2]))
            y_pred = color_label(y_pred, label_colors)
            print(nb_frames, y_pred.shape, y_pred.dtype)

            # Write the frame into the file 'output.avi'
            out.write(y_pred)

        cap.release()
        out.release()

if __name__ == '__main__':
    main()
