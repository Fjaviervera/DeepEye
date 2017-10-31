import cv2
import numpy as np
import tensorflow as tf


class Network:
    def __init__(self, name, imgs, batch_size, is_training=True, reuse=None, layers=16, deep=2):
        self.imgs = imgs
        self.batch_size = batch_size
        self.is_training = is_training
        self.reuse = reuse
        self.conv_counter = 0
        self.atrous_counter = 0
        self.transpose_counter = 0
        self.stride_counter = 0
        self.conv1x1_counter = 0
        self.name = name
        self.layers = layers
        self.deep = deep
        self.output = self.build_net()

    def conv1x1(self, x, filters):
        self.conv1x1_counter += 1
        with tf.variable_scope('Last_conv1x1_%d' % (self.conv1x1_counter), reuse=self.reuse):
            print('Last_conv1x1_%d' % (self.conv1x1_counter))

            x = tf.layers.conv2d(x, filters, [1, 1], padding='SAME')

        return x

    def conv_unit(self, x, filters):
        self.conv_counter += 1

        with tf.variable_scope('conv_unit_%d' % (self.conv_counter), reuse=self.reuse):
            print('conv_unit_%d' % (self.conv_counter))

            x_orig = x

            x = tf.layers.conv2d(x, filters, [3, 3], padding='SAME')
            
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=0.99,
                                              renorm=True)
            x = tf.nn.relu(x, name='relu')

            x = tf.layers.conv2d(x, filters, [3, 3], padding='SAME')
            
            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=0.99,
                                              renorm=True)

            x = tf.nn.relu(x + x_orig, name='relu')

            return x

    def stride_unit(self, x, filters_in, filters_out):
        self.stride_counter += 1
        with tf.variable_scope('stride_unit_%d' % (self.stride_counter), reuse=self.reuse):
            print('stride_unit_%d' % (self.stride_counter))

            x_orig = x

            x = tf.layers.conv2d(x, filters_out, [3, 3], padding='SAME', strides=(2, 2))  # , strides=(2, 2)

            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=0.99,
                                              renorm=True)

            x = tf.nn.relu(x, name='relu')

            x = tf.layers.conv2d(x, filters_out, [3, 3], padding='SAME')  # , strides=(2, 2)

            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=0.99,
                                              renorm=True)

            with tf.variable_scope('sub_add_up'):
                x_orig = tf.layers.average_pooling2d(x_orig, [2, 2], [2, 2], padding='SAME')
                x_orig = self.conv1x1(x_orig, filters_out)

            x = tf.nn.relu(x + x_orig, name='relu')

            return x

    def atrous_unit(self, x, filters, dilatation):
        x_orig = x

        x = tf.layers.conv2d(x, filters, [3, 3], padding='SAME', dilation_rate=dilatation)
        
        x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=0.99,
                                          renorm=True)

        x = tf.nn.relu(x, name='relu')

        x = tf.layers.conv2d(x, filters, [3, 3], padding='SAME', dilation_rate=dilatation)
        
        x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=0.99,
                                          renorm=True)

        x = tf.nn.relu(x + x_orig, name='relu')

        return x

    def aspp(self, x, filters):
        self.atrous_counter += 1
        with tf.variable_scope('atrous_unit_%d' % (self.atrous_counter), reuse=self.reuse):
            print('atrous_unit_%d' % (self.atrous_counter))

            x_1 = self.atrous_unit(x, filters, 4)
            x_2 = self.atrous_unit(x, filters, 8)
            x_3 = self.atrous_unit(x, filters, 16)
            x_4 = self.conv_unit(x, filters)
            x_5 = tf.layers.max_pooling2d(x, [2, 2], [2, 2], padding='SAME')
            
            shape_orig = tf.shape(x)
            shape_pool = tf.shape(x_5)
            
            x_5 = tf.pad(x_5, [[0, 0], [shape_orig[1] - shape_pool[1], 0], [shape_orig[2] - shape_pool[2], 0], [0, 0]])

            x_5 = tf.reshape(x_5, tf.shape(x))

            x = tf.concat([x_1, x_2, x_3, x_4, x_5], 3)
            
            x = tf.layers.conv2d(x, filters, [1, 1], padding='SAME')

            x = tf.layers.batch_normalization(x, training=self.is_training, reuse=self.reuse, momentum=0.99,
                                              renorm=True)
            x = tf.nn.relu(x, name='relu')
            
            return x

    def build_net(self):
        print(self.name)
        with tf.variable_scope(self.name, reuse=self.reuse):
            
            x = tf.reshape(self.imgs, [1, tf.shape(self.imgs)[0], tf.shape(self.imgs)[1], 1])
            orig_shapes = [tf.shape(self.imgs)[0], tf.shape(self.imgs)[1]]

            filters_num = self.layers
            x = self.conv_unit(x, filters_num)
            x = self.stride_unit(x, filters_num, 2 * filters_num)
            x = self.conv_unit(x, 2 * filters_num)
            x = self.stride_unit(x, 2 * filters_num, 4 * filters_num)

            for i in range(self.deep):
                x = self.aspp(x, 4 * filters_num)

            x = self.conv1x1(x, 2)
            self.y_mlp = x
            
            x = tf.image.resize_bilinear(x, [orig_shapes[0], orig_shapes[1]])
            
            return x


class DeepEye:
    def __init__(self, deep=2, layers=16, model='models/default.ckpt'):

        self.sess = tf.Session()

        self.frame_input = tf.placeholder(tf.uint8, [288, 384])

        input_reshaped_casted = tf.cast(self.frame_input, tf.float32) * (1. / 255)

        deepupil_network = Network('Deep_eye', input_reshaped_casted, 1, is_training=False, reuse=False, deep=deep,
                                   layers=layers)

        saver = tf.train.Saver(max_to_keep=0)

        self.prob_mask = tf.nn.softmax(deepupil_network.output)

        saver.restore(self.sess, model)
        print("Model restored.")

    def blob_location(self, prob_mask):

        factor = prob_mask.size / (288.0 * 384.0)

        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = 100 * factor
        params.maxArea = 25000 * factor

        params.filterByConvexity = True
        params.minConvexity = 0.1

        detector = cv2.SimpleBlobDetector_create(params)

        found_blob = False
        prob = 0.5
        raw_img = prob_mask.copy()
        while found_blob == False:
            image = raw_img.copy()
            image[image < prob] = 0
            image[image > prob] = 1
            image = (image * 255).astype('uint8')
            image = 255 - image
            keypoints = detector.detect(image)

            if len(keypoints) > 0:

                blob_sizes = []
                for k in keypoints:
                    blob_sizes.append(k.size)
                detection = np.argmax(np.asarray(blob_sizes))
                out_coordenate = [int(keypoints[detection].pt[0]), int(keypoints[detection].pt[1])]
                found_blob = True
            else:
                out_coordenate = [0, 0]
                found_blob = False
                prob += -0.05
                if prob < 0.05:
                    found_blob = True

        return out_coordenate

    def run(self, frame):

        if frame.shape != (288, 384):

            orig_size = frame.shape

            frame = cv2.resize(frame, (384, 288), cv2.INTER_LINEAR)
            prob_mask = self.sess.run(
                self.prob_mask,
                feed_dict={self.frame_input: frame})

            prob_mask = cv2.resize(prob_mask[0, :, :, 0], (orig_size[1], orig_size[0]), cv2.INTER_LINEAR)
        else:

            prob_mask = self.sess.run(
                self.prob_mask,
                feed_dict={self.frame_input: frame})

            prob_mask = prob_mask[0:, :, 0]
        return self.blob_location(prob_mask)

    def restart_tracker(self):

        tf.reset_default_graph()
        # plt.close('all')
        self.sess.close()
