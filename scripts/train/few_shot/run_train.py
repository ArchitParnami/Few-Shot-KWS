import argparse

from train import main

parser = argparse.ArgumentParser(description='Train prototypical networks')

# data args
default_dataset = 'omniglot'
parser.add_argument('--data.dataset', type=str, default=default_dataset, metavar='DS',
                    help="data set name (default: {:s})".format(default_dataset))
default_split = 'vinyals'
parser.add_argument('--data.split', type=str, default=default_split, metavar='SP',
                    help="split name (default: {:s})".format(default_split))
parser.add_argument('--data.way', type=int, default=60, metavar='WAY',
                    help="number of classes per episode (default: 60)")
parser.add_argument('--data.shot', type=int, default=5, metavar='SHOT',
                    help="number of support examples per class (default: 5)")
parser.add_argument('--data.query', type=int, default=5, metavar='QUERY',
                    help="number of query examples per class (default: 5)")
parser.add_argument('--data.test_way', type=int, default=5, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as data.way (default: 5)")
parser.add_argument('--data.test_shot', type=int, default=0, metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=15, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as data.query (default: 15)")
parser.add_argument('--data.train_episodes', type=int, default=100, metavar='NTRAIN',
                    help="number of train episodes per epoch (default: 100)")
parser.add_argument('--data.test_episodes', type=int, default=100, metavar='NTEST',
                    help="number of test episodes per epoch (default: 100)")
parser.add_argument('--data.trainval', action='store_true', help="run in train+validation mode (default: False)")
parser.add_argument('--data.sequential', action='store_true', help="use sequential sampler instead of episodic (default: False)")
parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")

# model args
default_model_name = 'protonet_conv'
default_encoding = 'C64'
parser.add_argument('--model.model_name', type=str, default=default_model_name, metavar='MODELNAME',
                    help="model name (default: {:s})".format(default_model_name))
parser.add_argument('--model.x_dim', type=str, default='1,28,28', metavar='XDIM',
                    help="dimensionality of input images (default: '1,28,28')")
parser.add_argument('--model.hid_dim', type=int, default=64, metavar='HIDDIM',
                    help="dimensionality of hidden layers (default: 64)")
parser.add_argument('--model.z_dim', type=int, default=64, metavar='ZDIM',
                    help="dimensionality of input images (default: 64)")
parser.add_argument('--model.encoding', type=str, default=default_encoding, metavar='MODELENC',
                    help="model encoding (default: {:s})".format(default_encoding))
# train args
parser.add_argument('--train.epochs', type=int, default=10000, metavar='NEPOCHS',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--train.optim_method', type=str, default='Adam', metavar='OPTIM',
                    help='optimization method (default: Adam)')
parser.add_argument('--train.learning_rate', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--train.decay_every', type=int, default=20, metavar='LRDECAY',
                    help='number of epochs after which to decay the learning rate')
default_weight_decay = 0.0
parser.add_argument('--train.weight_decay', type=float, default=default_weight_decay, metavar='WD',
                    help="weight decay (default: {:f})".format(default_weight_decay))
parser.add_argument('--train.patience', type=int, default=200, metavar='PATIENCE',
                    help='number of epochs to wait before validation improvement (default: 1000)')

# log args
default_fields = 'loss,acc'
parser.add_argument('--log.fields', type=str, default=default_fields, metavar='FIELDS',
                    help="fields to monitor during training (default: {:s})".format(default_fields))
default_exp_dir = 'results'
parser.add_argument('--log.exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR',
                    help="directory where experiments should be saved (default: {:s})".format(default_exp_dir))

# speech data args
parser.add_argument('--speech.include_background', action='store_true', help="mix background noise with samples (default: False)")
parser.add_argument('--speech.include_silence', action='store_true', help="one of the classes out of n should be silence (default: False)")
parser.add_argument('--speech.include_unknown', action='store_true', help="one of the classes out of n should be unknown (default: False)")
parser.add_argument('--speech.sample_rate', type=int, default=16000, help='desired sampling rate of the input')
parser.add_argument('--speech.clip_duration', type=int, default=1000, help='clip duration in milliseconds')
parser.add_argument('--speech.window_size', type=int, default=40)
parser.add_argument('--speech.window_stride', type=int,default=20)
parser.add_argument('--speech.num_features', type=int, default=40, help='Number of mfcc features to extract')
parser.add_argument('--speech.time_shift', type=int, default=100, help='time shift the audio in milliseconds')
parser.add_argument('--speech.bg_volume', type=float, default=0.1, help='background volumen to mix in between 0 and 1')
parser.add_argument('--speech.bg_frequency', type=float, default=1.0, help='Amount of samples that should be mixed with background noise (between 0 and 1)')
parser.add_argument('--speech.num_silence', type=int, default=1000, help='Number of silence samples to generate')
parser.add_argument('--speech.foreground_volume', type=float, default=1)
args = vars(parser.parse_args())

main(args)
