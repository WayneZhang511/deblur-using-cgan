from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
import numpy as np

from ops import *
from utils import *

class pix2pix:
    model_name = 'pix2pix'
    
    def __init__(self, 
                 batch_size=1, 
                 input_width=256, 
                 input_height=256, 
                 input_channels=3, 
                 output_channels=3, 
                 df_dim=64, 
                 gf_dim=64, 
                 L1_lambda=100,
                 checkpoint_dir=None,
                 checkpoint_name=None,
                 dataset_name=None,
                 dataset_dir=None,
                 sample_dir=None,
                 test_dir=None,
                 sess=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        
        self.batch_size = batch_size
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.L1_lambda = L1_lambda
        self.sess = sess
        self.is_grayscale = (input_channels == 1)
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_name == '':
            self.checkpoint_name = self.model_dir
        else:
            self.checkpoint_name = checkpoint_name
        self.dataset_dir = dataset_dir
        self.sample_dir = sample_dir
        self.test_dir = test_dir
        self.build_model()
    
    def build_model(self):
        # input
        self.input_A = tf.placeholder(tf.float32,
                                     [self.batch_size, self.input_height, self.input_width, self.input_channels],
                                     name='input_A')
        self.input_B = tf.placeholder(tf.float32,
                                     [self.batch_size, self.input_height, self.input_width, self.output_channels],
                                     name='input_B')
        self.input_AB = tf.concat([self.input_A, self.input_B], 3)
        assert self.input_AB.get_shape().as_list() == [self.batch_size, self.input_height, self.input_width, self.input_channels + self.output_channels]
        
        # feed real pair in
        self.D_real_logits = self.discriminator(self.input_AB, reuse=False)
        
        # generate
        self.fake_B = self.generator(self.input_A, reuse=False)
        self.fake_AB = tf.concat([self.input_A, self.fake_B], 3)
        
        # feed fake pair in
        self.D_fake_logits = self.discriminator(self.fake_AB, reuse=True)
        
        # calculate discriminator loss
        # real images: encourage ones
        self.D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits,labels=tf.ones_like(self.D_real_logits)))
        # fake images: encourage zeros
        self.D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits,labels=tf.zeros_like(self.D_fake_logits)))
        # sum up
        self.d_loss = self.D_real_loss + self.D_fake_loss
        
        # calculate generator loss
        # L1 loss
        self.L1_loss = tf.reduce_mean(tf.abs(self.fake_B - self.input_B))
        # fake images: encourage ones
        self.G_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.ones_like(self.D_fake_logits)))
        # sum up
        self.g_loss = self.L1_lambda * self.L1_loss + self.G_adv_loss
            
        # trainable variables
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        
        # tensorboard
        self.d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
        self.L1_loss_summary = tf.summary.scalar('L1_loss', self.L1_loss)
        self.g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
        self.summaries = tf.summary.merge_all()
        self.train_summary_writer = tf.summary.FileWriter('logs/train', self.sess.graph)
        self.val_summary_writer = tf.summary.FileWriter('logs/val', self.sess.graph)

        # save model
        self.saver = tf.train.Saver()
    
    def generator(self, image, reuse=False):
        with tf.variable_scope('generator') as scope:
            # TODO: reuse???
            if reuse:
                scope.resue_variables()
            
            with slim.arg_scope([slim.conv2d_transpose, slim.conv2d],
                               weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                               activation_fn=None,
                               normalizer_fn=slim.batch_norm,
                               padding='SAME'):
                
                # decoder
                conv1 = slim.conv2d(image, self.gf_dim, [5,5], stride=2, normalizer_fn=None, scope='g_conv1')
                conv2 = slim.conv2d(leaky_relu(conv1), self.gf_dim * 2, [5,5], stride=2, scope='g_conv2')
                conv3 = slim.conv2d(leaky_relu(conv2), self.gf_dim * 4, [5,5], stride=2, scope='g_conv3')
                conv4 = slim.conv2d(leaky_relu(conv3), self.gf_dim * 8, [5,5], stride=2, scope='g_conv4')
                conv5 = slim.conv2d(leaky_relu(conv4), self.gf_dim * 8, [5,5], stride=2, scope='g_conv5')
                conv6 = slim.conv2d(leaky_relu(conv5), self.gf_dim * 8, [5,5], stride=2, scope='g_conv6')
                conv7 = slim.conv2d(leaky_relu(conv6), self.gf_dim * 8, [5,5], stride=2, scope='g_conv7')
                conv8 = slim.conv2d(leaky_relu(conv7), self.gf_dim * 8, [5,5], stride=2, activation_fn=None, scope='g_conv8')
                
                # encoder
                dconv1 = slim.conv2d_transpose(tf.nn.relu(conv8), self.gf_dim * 8, [5,5], stride=2, activation_fn=None, scope='g_dconv1')
                dconv1 = tf.nn.dropout(dconv1, 0.5)
                dconv1 = tf.concat([dconv1, conv7], 3)
                
                dconv2 = slim.conv2d_transpose(tf.nn.relu(dconv1), self.gf_dim * 8, [5,5], stride=2, activation_fn=None, scope='g_dconv2')
                dconv2 = tf.nn.dropout(dconv2, 0.5)
                dconv2 = tf.concat([dconv2, conv6], 3)
                
                dconv3 = slim.conv2d_transpose(tf.nn.relu(dconv2), self.gf_dim * 8, [5,5], stride=2, activation_fn=None, scope='g_dconv3')
                dconv3 = tf.nn.dropout(dconv3, 0.5)
                dconv3 = tf.concat([dconv3, conv5], 3)
                
                dconv4 = slim.conv2d_transpose(tf.nn.relu(dconv3), self.gf_dim * 8, [5,5], stride=2, activation_fn=None, scope='g_dconv4')
                # dconv4 = tf.nn.dropout(dconv4, 0.5)
                dconv4 = tf.concat([dconv4, conv4], 3)
                
                dconv5 = slim.conv2d_transpose(tf.nn.relu(dconv4), self.gf_dim * 4, [5,5], stride=2, activation_fn=None, scope='g_dconv5')
                # dconv5 = tf.nn.dropout(dconv5, 0.5)
                dconv5 = tf.concat([dconv5, conv3], 3)
                
                dconv6 = slim.conv2d_transpose(tf.nn.relu(dconv5), self.gf_dim * 2, [5,5], stride=2, activation_fn=None, scope='g_dconv6')
                # dconv6 = tf.nn.dropout(dconv6, 0.5)
                dconv6 = tf.concat([dconv6, conv2], 3)
                
                dconv7 = slim.conv2d_transpose(tf.nn.relu(dconv6), self.gf_dim, [5,5], stride=2, activation_fn=None, scope='g_dconv7')
                # dconv7 = tf.nn.dropout(dconv7, 0.5)
                dconv7 = tf.concat([dconv7, conv1], 3)
                
                output = slim.conv2d_transpose(tf.nn.relu(dconv7), self.output_channels, [5,5], stride=2, normalizer_fn=None, activation_fn=tf.nn.tanh, scope='g_out')
                
                return output
            
    def discriminator(self, image, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
                
            with slim.arg_scope([slim.conv2d],
                               weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                               activation_fn=None,
                               normalizer_fn=slim.batch_norm,
                               padding='SAME'):
                # 256 -> 128
                conv1 = slim.conv2d(image, self.df_dim, [5,5], stride=2, normalizer_fn=None, scope='d_conv1')
                
                # 128 -> 64
                conv2 = slim.conv2d(leaky_relu(conv1), self.df_dim * 2, [5,5], stride=2, scope='d_conv2')
                
                # 64 -> 32
                conv3 = slim.conv2d(leaky_relu(conv2), self.df_dim * 4, [5,5], stride=2, scope='d_conv3')
                
                # 32 -> 16
                conv4 = slim.conv2d(leaky_relu(conv3), self.df_dim * 8, [5,5], stride=2, scope='d_conv4')
                
                # flatten
                conv4_flat = tf.reshape(conv4, [self.batch_size, -1])
                fc1 = slim.fully_connected(conv4_flat, 1, normalizer_fn=None, activation_fn=None, scope='d_fc1')
                
                return fc1
                
                
    def train(self, args):
        # set optimizer
        d_optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=args.beta1, beta2=args.beta2) \
                            .minimize(self.d_loss, var_list=self.d_vars)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=args.beta1, beta2=args.beta2) \
                            .minimize(self.g_loss, var_list=self.g_vars)
        
        # add additional options to trace the session execution
        # self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # self.run_metadata = tf.RunMetadata()
        
        print('initialization...')
        # initialization
        # init_op = tf.global_variables_initializer().run()
        # self.sess.run(init_op)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        counter = 0
        
        print('perpare data...')
        # perpare data
        data_list = glob(os.path.join(args.dataset_dir, args.dataset_name, args.phase, '*.png'))
        print(data_list)
        print(os.path.join(args.dataset_dir, args.dataset_name, args.phase, '*.png'))
        print(len(data_list))
        batch_idxs = int(len(data_list) / self.batch_size)
        #print(batch_idxs)
        # load checkpoint
        check_bool, counter = self.load_model(self.checkpoint_dir, self.checkpoint_name)
        if check_bool:
            print('Load model successfully')
        else:
            print('Fail to load model')
        counter += 1
        
        # training
        print('start training...')
        start_time = time.time()
        for epoch in range(args.epoch):
            for idx in range(batch_idxs):
                # read images
                batch_files = data_list[idx * self.batch_size: (idx + 1) * self.batch_size]
                batch = [get_image(batch_file) for batch_file in batch_files]
                #print(len(batch))
                #print(batch.shape[0])
                # deal with grayscale image
                if self.is_grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:,:,:,None]
                else:
                    batch_images = np.array(batch).astype(np.float32)
                #print(batch_images.shape) 
                # B to A
                input_B = batch_images[:, :, :self.input_width, :]
                input_A = batch_images[:, :, self.input_width:, :]
                
                # feed in data
                _, d_loss, summaries = self.sess.run([d_optimizer, self.d_loss, self.summaries],
                                                    feed_dict={self.input_A:input_A, self.input_B:input_B})
                _, g_loss, L1_loss, summaries = self.sess.run([g_optimizer, self.g_loss, self.L1_loss, self.summaries],
                                                             feed_dict={self.input_A:input_A, self.input_B:input_B})
                #print('writing logs...') 
                # update summary
                counter += 1
                end_time = time.time()
                total_time = end_time - start_time
                print('epoch{}[{}/{}]:phase:{}, total_time:{:.4f}, d_loss:{:.4f}, g_loss:{:.4f}, l1_loss:{:.4f}'.format(epoch, idx, batch_idxs, args.phase, total_time, d_loss, g_loss, self.L1_lambda*L1_loss))
                self.train_summary_writer.add_summary(summaries, global_step=counter)
                
                # sample and save checkpoint
                if np.mod(counter, 10) == 0:
                    self.sample(args.sample_dir, epoch, idx, counter)
                if np.mod(counter, 500) == 0:
                    self.save_model(self.checkpoint_dir, counter)

                # Create the Timeline object, and write it to a json file
                # fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
                # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                # with open('timeline_02_step_%d.json' % counter, 'w') as f:
                #     f.write(chrome_trace)
                    
    def test(self, args):
        # prepare data
        data_list = glob(os.path.join(args.dataset_dir, args.dataset_name, args.phase, '*.png'))
        # batch size is 1
        batch_idxs = len(data_list)
        # batch_idxs = int(len(data_list) / self.batch_size)
        
        # load model
        counter = 0
        check_bool, counter = self.load_model(self.checkpoint_dir, self.checkpoint_name)
        if check_bool:
            print('Load model successfully')
        else:
            print('Fail to laod model')
            return 
        
        for idx in range(batch_idxs):
            # read images
            batch_files = data_list[idx : idx + 1]
            batch = [get_image(batch_file) for batch_file in batch_files]

            # deal with grayscale image
            if self.is_grayscale:
                batch_images = np.array(batch).astype(np.float32)[:,:,:,None]
            else:
                batch_images = np.array(batch).astype(np.float32)
            # split B if any
            # B to A
            input_B = batch_images[:, :, :self.input_width, :]
            input_A = batch_images[:, :, self.input_width:, :]
            
            # run model
            sample_B = self.sess.run(self.fake_B,
                                    feed_dict={self.input_A:input_A})
            
            # save results
            sample = np.concatenate([input_A, input_B, sample_B], 2)
            save_images(sample, [1,1], '{}/{}_{:04d}.png'.format(self.test_dir, self.dataset_name, idx))
            print('testing:{}'.format(idx))
    
    # sample for display when doing training
    def sample(self, sample_dir, epoch, idx, counter):
        print("")
        input_A, input_B = self.load_sample()
        sample_B, d_loss, g_loss, L1_loss, summaries = self.sess.run([self.fake_B, self.d_loss, self.g_loss, self.L1_loss, self.summaries],
                                feed_dict={self.input_A:input_A, self.input_B:input_B})
        print('sampling: d_loss:{:.4f}, g_loss:{:.4f}, l1_loss:{:.4f}'.format(d_loss, g_loss, self.L1_lambda*L1_loss))
        self.val_summary_writer.add_summary(summaries, global_step=counter)

        if np.mod(counter, 100) == 0:
            sample = np.concatenate([input_A, input_B, sample_B], 2)
            save_images(sample, [self.batch_size,1], '{}/{}_{:04d}_{:04d}.png'.format(sample_dir,self.dataset_name, epoch, idx))        
    
    def load_sample(self):
        # prepare data
        batch_files = np.random.choice(glob(os.path.join(self.dataset_dir, self.dataset_name, 'val', '*.png')), self.batch_size)

        # read images
        batch = [get_image(batch_file) for batch_file in batch_files]
        if self.is_grayscale:
            batch_images = np.array(batch).astype(np.float32)[:,:,:,None]
        else:
            batch_images = np.array(batch).astype(np.float32)
        
        # B to A
        input_B = batch_images[:,:,:self.input_width,:]
        input_A = batch_images[:,:,self.input_width:,:]
        
        return input_A, input_B
    
    # image size: 1280*720
    def deblur(self, args):
        data_list = glob(os.path.join(args.dataset_dir, args.dataset_name, '*.png'))

        counter = 0
        check_bool, counter = self.load_model(self.checkpoint_dir, self.checkpoint_name)

        if check_bool:
            print('Load model successfully')
        else:
            print('Fail to load model')
            return

        for path in data_list:
            image_name = os.path.basename(path)
            print('Deblurring image {}'.format(image_name))
            image = get_image(path)

            canvas = np.zeros_like(image)

            input_A = generate_patches(image)

            output_B = self.sess.run(self.fake_B,
                        feed_dict={self.input_A:input_A})

            output = merge_images(canvas, output_B)

            
            output_dir = os.path.join(args.dataset_dir, 'deblurred-' + args.dataset_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            scipy.misc.imsave('{}/{}'.format(output_dir, image_name), output)

    # save model            
    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size)

    def save_model(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)
        
    def load_model(self, checkpoint_dir, checkpoint_name):
        """Load checkpoint"""
        import re
        
        print("reading checkpoint...")
        checkpoint_dir = os.path.join(checkpoint_dir, checkpoint_name, self.model_name)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
