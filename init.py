import numpy as np
import argparse
from train import *
from test import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        type=str,
                        help='The path to the pretrained cscapes model')

    parser.add_argument('-i', '--image-path',
                        type=str,
                        help='The path to the image to perform semantic segmentation')

    parser.add_argument('-rh', '--resize-height',
                        type=int,
                        default=1024,
                        help='The height for the resized image')

    parser.add_argument('-rw', '--resize-width',
                        type=int,
                        default=2048,
                        help='The width for the resized image')

    parser.add_argument('-lr', '--learning-rate',
                        type=float,
                        default=1e-5,
                        help='The learning rate')

    parser.add_argument('-bs', '--batch-size',
                        type=int,
                        default=2,
                        help='The batch size')

    parser.add_argument('-wd', '--weight-decay',
                        type=float,
                        default=1e-4,
                        help='The weight decay')

    parser.add_argument('-c', '--constant',
                        type=float,
                        default=1.02,
                        help='The constant used for calculating the class weights')

    parser.add_argument('-e', '--epochs',
                        type=int,
                        default=100,
                        help='The number of epochs')

    parser.add_argument('-nc', '--num-classes',
                        type=int,
                        default=None,
                        help='Number of classes (required for train mode, optional for test mode where it can be inferred from checkpoint)')

    parser.add_argument('-se', '--save-every',
                        type=int,
                        default=10,
                        help='The number of epochs after which to save a model')

    parser.add_argument('-iptr', '--input-path-train',
                        type=str,
                        help='The path to the input dataset')

    parser.add_argument('-lptr', '--label-path-train',
                        type=str,
                        help='The path to the label dataset')

    parser.add_argument('-ipv', '--input-path-val',
                        type=str,
                        help='The path to the input dataset')

    parser.add_argument('-lpv', '--label-path-val',
                        type=str,
                        help='The path to the label dataset')

    parser.add_argument('-iptt', '--input-path-test',
                        type=str,
                        help='The path to the input dataset')

    parser.add_argument('-lptt', '--label-path-test',
                        type=str,
                        help='The path to the label dataset')

    parser.add_argument('--out-dir',
                        type=str,
                        default='preds_val',
                        help='Directory to save prediction/evaluation outputs')

    parser.add_argument('--num-samples',
                        type=int,
                        default=5,
                        help='Number of random qualitative samples to save in test mode')

    parser.add_argument('-pe', '--print-every',
                        type=int,
                        default=1,
                        help='The number of epochs after which to print the training loss')

    parser.add_argument('-ee', '--eval-every',
                        type=int,
                        default=10,
                        help='The number of epochs after which to print the validation loss')

    parser.add_argument('--cuda',
                        type=bool,
                        default=False,
                        help='Whether to use cuda or not')

    parser.add_argument('--mode',
                        choices=['train', 'test'],
                        default='train',
                        help='Whether to train or test')
    
    parser.add_argument('-dt', '--dtype',
                        choices=['cityscapes', 'pascal'],
                        default='pascal',
                        help='specify the dataset you are using')
    
    parser.add_argument('--scheduler',
                        type=str,
                        default='poly',
                        help='Learning-rate scheduler to use: poly or none')

    parser.add_argument('--grad-accum-steps',
                        type=int,
                        default=1,
                        help='Number of gradient accumulation steps')

    parser.add_argument('--sync-bn',
                        type=str,
                        default='False',
                        help='Use SyncBatchNorm when running distributed training')

    parser.add_argument('--optimizer',
                        choices=['sgd', 'adam'],
                        default='sgd',
                        help='Optimizer: sgd or adam')
    
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='Momentum for SGD optimizer')
    
    parser.add_argument('--use-class-weights',
                        type=bool,
                        default=True,
                        help='Whether to weight loss by class frequency')
    
    parser.add_argument('--early-stop-patience',
                        type=int,
                        default=12,
                        help='Early stopping patience (in epochs) — reduced for better convergence detection')
    
    parser.add_argument('--augment',
                        type=bool,
                        default=True,
                        help='Whether to use data augmentation')

    parser.add_argument('--save',
                        type=bool,
                        default=True,
                        help='Save the segmented image when predicting')

    parser.add_argument('--multi-scale-test',
                        type=str,
                        default='False',
                        help='Use multi-scale inference during testing')

    parser.add_argument('--flip-test',
                        type=str,
                        default='False',
                        help='Use horizontal flip inference during testing')

    parser.add_argument('--tta-scales',
                        type=float,
                        nargs='+',
                        default=[0.75, 1.0, 1.25],
                        help='Scales to use for multi-scale testing')

    parser.add_argument('--imagenet-normalize',
                        type=str,
                        default='False',
                        help='Use ImageNet mean/std normalization at test time (default: False to match training preprocessing)')

    parser.add_argument('--backbone',
                        choices=['resnet50', 'resnet101'],
                        default='resnet50',
                        help='Backbone for DeepLabv3 encoder')

    parser.add_argument('--output-stride',
                        type=int,
                        default=16,
                        help='Backbone output stride for DeepLabv3 (4, 8, or 16)')

    parser.add_argument('--pretrained-backbone',
                        type=str,
                        default='True',
                        help='Use ImageNet pretrained backbone weights')

    FLAGS, unparsed = parser.parse_known_args()

    FLAGS.cuda = torch.device('cuda:0' if torch.cuda.is_available() and FLAGS.cuda \
                               else 'cpu')
    
    print ('[INFO]Arguments read successfully!')

    if FLAGS.mode.lower() == 'train':
        print ('[INFO]Train Mode.')

        if FLAGS.num_classes is None:
            raise RuntimeError('Error: --num-classes is required for training mode')
        if FLAGS.input_path_train == None or FLAGS.input_path_val == None:
            raise ('Error: Kindly provide the path to the dataset')

        train(FLAGS)

    elif FLAGS.mode.lower() == 'test':
        print ('[INFO]Predict Mode.')
        predict(FLAGS)
    else:
        raise RuntimeError('Unknown mode passed. \n Mode passed should be either \
                            of "train" or "test"')
