import argparse
import os
import scipy.misc
import numpy as np

from model import pix2pix
import tensorflow as tf

parser = argparse.ArgumentParser()
# parser.add_argument('--lambd', type=float, default=100, help='weights of l1 loss')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2')
parser.add_argument('--dataset_name', type=str, default='gopro', help='dataset name')
parser.add_argument('--dataset_dir', type=str, default='.', help='dataset directory')
# parser.add_argument('--is_training', type=str, default='False', help='training or testing')
# parser.add_argument('--is_testing', type=str, default='False', help='training or testing')
parser.add_argument('--epoch', type=int, default=200, help='training epoches')
parser.add_argument('--phase', type=str, default='train', help='indicating training or testing phase')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='checkpoint directory')
parser.add_argument('--checkpoint_name', type=str, default='', help='checkpoint name to use')
parser.add_argument('--test_dir', default='test', type=str, help='testing directory')
parser.add_argument('--sample_dir', type=str, default='sample', help='sample directory')
parser.add_argument('--logs_dir', type=str, default='logs', help='log directory')


## TODO: modify the directory structure: model -> checkpoint, sample, log
## TODO: decouple the checkpoint names and dataset names of train and test
def check_dir():
    if not os.path.exists('sample'):
        os.mkdir('sample')
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.exists('test'):
        os.mkdir('test')

def main(_):
    check_dir()
    args = parser.parse_args()
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    # config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True, gpu_options=gpu_options)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = pix2pix(sess=sess, batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir, dataset_name=args.dataset_name,
            dataset_dir=args.dataset_dir, sample_dir=args.sample_dir,
            test_dir=args.test_dir, checkpoint_name=args.checkpoint_name)
        if args.phase == 'train':
            model.train(args)
        else:
            model.test(args)

if __name__ == '__main__':
    tf.app.run()



