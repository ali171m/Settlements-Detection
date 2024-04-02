#from google.colab import drive
#drive.mount('/gdrive')

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import cv2
import os, sys
import numpy as np
import math
from datetime import datetime
import time
from PIL import Image
from math import ceil
from tensorflow.python.ops import gen_nn_ops
from skimage import io
# modules


IMAGE_CHANNELS=1
NUM_CLASSES=2
BATCH_SIZE=4
IMAGE_HEIGHT=256
IMAGE_WIDTH=256
LEARNING_RATE = 0.001      # Initial learning rate.


class Dataset:
    def __init__(self, folder, include_hair=False):
        
        self.batch_size = BATCH_SIZE
        
        train_files= os.listdir(os.path.join(folder, 'inputs'))
        train_targetfiles=os.listdir(os.path.join(folder, 'targets_jpg'))
        test_files = train_files
        test_targetfiles=train_targetfiles

        self.train_inputs, self.train_targets = self.file_paths_to_images(folder, train_files,train_targetfiles)
   
        self.test_inputs, self.test_targets = self.file_paths_to_images(folder, test_files,test_targetfiles, True)
        self.pointer=0
      

    def file_paths_to_images(self, folder, files_list,files_list2, verbose=False):
        inputs = []
        targets = []

        for file in files_list:
            input_image = os.path.join(folder, 'inputs', file)
            
            
            test_image = np.array(cv2.imread(input_image, 0))  # load grayscale
            
            # test_image = np.multiply(test_image, 1.0 / 255)
            inputs.append(test_image)
            
        for file2 in files_list2:
            
            target_image = os.path.join(folder, 'targets_jpg', file2)
            
            target_image = cv2.imread(target_image,0)
            
            target_image = np.array(target_image)
            target_image = np.multiply(target_image, 1.0 / 255)
            #target_image = cv2.threshold(target_image, 127, 1, cv2.THRESH_BINARY)[1]
           
            targets.append(target_image)
        
        return inputs, targets

    def train_valid_test_split(self, X, ratio=None):
        if ratio is None:
            ratio = (1, 0, 0)

        N = len(X)
        return (
            X[:int(ceil(N * ratio[0]))],
            X[int(ceil(N * ratio[0])): int(ceil(N * ratio[0] + N * ratio[1]))],
            X[int(ceil(N * ratio[0] + N * ratio[1])):]
        )

    def num_batches_in_epoch(self):
        
        return int(math.floor(len(self.train_inputs) / self.batch_size))

    def reset_batch_pointer(self):
        permutation = np.random.permutation(len(self.train_inputs))
        self.train_inputs = [self.train_inputs[i] for i in permutation]
        self.train_targets = [self.train_targets[i] for i in permutation]

        self.pointer = 0

    def next_batch(self):
        inputs = []
        targets = []
        for i in range(BATCH_SIZE):
            inputs.append(np.array(self.train_inputs[self.pointer + i]))
            targets.append(np.array(self.train_targets[self.pointer + i]))

        self.pointer += self.batch_size
        #return inputs,targets
        return np.array(inputs), np.array(targets)

    @property
    def test_set(self):
        return np.array(self.test_inputs, dtype=np.uint8), np.array(self.test_targets, dtype=np.uint8)

def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/gpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var
def orthogonal_initializer(scale = 1.1):
    
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer
def _variable_with_weight_decay(name, shape, initializer, wd):
  var = _variable_on_cpu(
      name,
      shape,
      initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var
def conv_layer_with_bn(inputT, shape, train_phase, activation=True, name=None):
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]
    with tf.variable_scope(name) as scope:
      kernel = _variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
      conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
      bias = tf.nn.bias_add(conv, biases)
      if activation is True:
        conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
      else:
        conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out

def get_deconv_filter(f_shape):

  width = f_shape[0]
  heigh = f_shape[0]
  f = ceil(width/2.0)
  c = (2 * f - 1 - f % 2) / (2.0 * f)
  bilinear = np.zeros([f_shape[0], f_shape[1]])
  for x in range(width):
      for y in range(heigh):
          value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
          bilinear[x, y] = value
  weights = np.zeros(f_shape)
  for i in range(f_shape[2]):
      weights[:, :, i, i] = bilinear

  init = tf.constant_initializer(value=weights,
                                 dtype=tf.float32)
  return tf.get_variable(name="up_filter", initializer=init,
                         shape=weights.shape)

def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):
  # output_shape = [b, w, h, c]
  # sess_temp = tf.InteractiveSession()
  sess_temp = tf.global_variables_initializer()
  strides = [1, stride, stride, 1]
  with tf.variable_scope(name):
    weights = get_deconv_filter(f_shape)
    deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
  return deconv

def batch_norm_layer(inputT, is_training, scope):
  return tf.cond(is_training,
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                           center=False, updates_collections=None, scope=scope+"_bn"),
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                           updates_collections=None, center=False, scope=scope+"_bn", reuse = True))



def inference(images, labels,  phase_train):
    # norm1
    norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
                name='norm1')
    # conv1
    conv1 = conv_layer_with_bn(norm1, [7, 7, images.get_shape().as_list()[3], 64], phase_train, name="conv1")
    # pool1
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # conv2
    conv2 = conv_layer_with_bn(pool1, [7, 7, 64, 64], phase_train, name="conv2")

    # pool2
    pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # conv3
    conv3 = conv_layer_with_bn(pool2, [7, 7, 64, 64], phase_train, name="conv3")

    # pool3
    pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    #conv4
    conv4 = conv_layer_with_bn(pool3, [7, 7, 64, 64], phase_train, name="conv4")

    # pool4
    pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    # conv5
    conv5 = conv_layer_with_bn(pool4, [7, 7, 64, 64], phase_train, name="conv5")

    # pool5
    pool5, pool5_indices = tf.nn.max_pool_with_argmax(conv5, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool5')
    #conv6
    conv6 = conv_layer_with_bn(pool5, [7, 7, 64, 64], phase_train, name="conv5")

    # pool6
    pool6, pool6_indices = tf.nn.max_pool_with_argmax(conv6, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool6')
    
    """ End of encoder """
    """ start upsample """
    # upsample6
    # Need to change when using different dataset out_w, out_h
    upsample6 = deconv_layer(pool6, [2, 2, 64, 64], [BATCH_SIZE, 8, 8, 64], 2, "up6")
    # decode 6
    conv_decode6 = conv_layer_with_bn(upsample6, [7, 7, 64, 64], phase_train, False, name="conv_decode6")
    
    # upsample5
    upsample5 = deconv_layer(conv_decode6, [2, 2, 64, 64], [BATCH_SIZE, 16, 16, 64], 2, "up5")
    # decode 5
    conv_decode5 = conv_layer_with_bn(upsample5, [7, 7, 64, 64], phase_train, False, name="conv_decode5")

    # upsample 4
    upsample4= deconv_layer( conv_decode5 , [2, 2, 64, 64], [BATCH_SIZE, 32, 32, 64], 2, "up4")
    # decode 4
    conv_decode4 = conv_layer_with_bn(upsample4, [7, 7, 64, 64], phase_train, False, name="conv_decode4")

    # upsample3
    upsample3= deconv_layer(conv_decode4, [2, 2, 64, 64], [BATCH_SIZE, 64, 64, 64], 2, "up3")
    # decode 3
    conv_decode3 = conv_layer_with_bn(upsample3, [7, 7, 64, 64], phase_train, False, name="conv_decode3")

    # upsample2
    upsample2= deconv_layer(conv_decode3, [2, 2, 64, 64], [BATCH_SIZE, 128, 128, 64], 2, "up2")
    # decode4
    conv_decode2 = conv_layer_with_bn(upsample2, [7, 7, 64, 64], phase_train, False, name="conv_decode2")

    # upsample1
    upsample1= deconv_layer(conv_decode2, [2, 2, 64, 64], [BATCH_SIZE, 256, 256, 64], 2, "up1")
    # decode1
    conv_decode1 = conv_layer_with_bn(upsample1, [7, 7, 64, 64], phase_train, False, name="conv_decode1")
    print((tf.sigmoid(conv_decode1)).shape)
    
    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[1, 1, 64, 1],
                                           initializer=msra_initializer(1, 64),
                                           wd=0.0005)
      conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [1], tf.constant_initializer(0.0))
      conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier

    return  logit



def training():
  
    #read training image
    dataset = Dataset(folder='/gdrive/My Drive/Colab Notebooks/data{}_{}old'.format(IMAGE_HEIGHT, IMAGE_WIDTH), include_hair=False)
    batch_inputs, batch_targets = dataset.next_batch()
    with tf.Graph().as_default():

        train_data_node = tf.placeholder( tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])

        train_labels_node = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

        phase_train = tf.placeholder(tf.bool, name='phase_train')

        global_step = tf.Variable(0, trainable=False)

        # Build a Graph that computes the logits predictions from the inference model.
        eval_prediction = inference(train_data_node, train_labels_node,  phase_train)
        segmentation_result = tf.sigmoid(eval_prediction)
       
        labels=train_labels_node
        print(labels.shape)
        print(eval_prediction.shape)
        
        cost = tf.sqrt(tf.reduce_mean(tf.square(segmentation_result-labels)))
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
        with tf.name_scope('accuracy'):
            argmax_probs = tf.round(segmentation_result)  # 0x1
            correct_pred = tf.cast(tf.equal(argmax_probs, labels), tf.float32)
            accuracy = tf.reduce_mean(correct_pred)
        
    
    
   
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            test_accuracies = []
            # Fit all training data
            n_epochs = 250
            global_start = time.time()
            for epoch_i in range(100):
                dataset.reset_batch_pointer()
                total_acc=0
                total_cost=0
                for batch_i in range(dataset.num_batches_in_epoch()):
                    
                    batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1

                    start = time.time()
                    
                    batch_inputs, batch_targets = dataset.next_batch()
                    batch_inputs = np.reshape(batch_inputs,(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
                    batch_targets = np.reshape(batch_targets,(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
                    feed_dict = {train_data_node: batch_inputs, train_labels_node:batch_targets,phase_train: True}
                  
                    
                    _, batch_cost= sess.run([train_op, cost], feed_dict=feed_dict)
                    correct,ll,batch_accuracy=sess.run([correct_pred,labels,accuracy],feed_dict=feed_dict)
                    total_acc += batch_accuracy  
                    total_cost +=batch_cost
                    
                    
                    #print("Best accuracy: {} acuracy {} in epoc {} batch{}".format(cost , test_accuracy,epoch_i,batch_i))
                   
                    end = time.time()
                
                avg_acc= total_acc/dataset.num_batches_in_epoch()
                avg_cost=total_cost/dataset.num_batches_in_epoch()
                print("avg cost: {} avg acuracy: {} in epoc: {}".format(avg_cost ,avg_acc, epoch_i))
                
                '''test_inputs, test_targets = dataset.test_set
                test_inputs, test_targets = test_inputs[:5], test_targets[:5]

                test_inputs = np.reshape(test_inputs, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
                test_targets = np.reshape(test_targets, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
                                      
                feed_dict = {train_data_node: test_inputs, train_labels_node: test_targets,phase_train: False}
                test_accuracy = sess.run([accuracy], feed_dict=feed_dict)

                
                test_accuracies.append((test_accuracy, batch_num))
                print("Accuracies in time: ", [test_accuracies[x][0] for x in range(len(test_accuracies))])
                max_acc = max(test_accuracies)  
                print("Best accuracy: {} in batch {}".format(max_acc[0], max_acc[1]))
                #print("Best accuracy: {} ".format(test_accuracy))
                print("Total time: {}".format(time.time() - global_start))'''

                  if test_accuracy >= max_acc[0]:
                  checkpoint_path = os.path.join('save', network.description, timestamp, 'model.ckpt')
                  saver.save(sess, checkpoint_path, global_step=batch_num)'''


if __name__ == '__main__':
    training()
