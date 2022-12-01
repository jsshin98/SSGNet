import argparse
import pprint
from collections import OrderedDict

class TestOptions:
    def __init__(self):
        parser = argparse.ArgumentParser()

        # ------------------
        # dataset parameters
        # ------------------
        parser.add_argument('--data-root', type=str, default='/shared/dataset/', metavar='D',
                            help='place to find (or download) data')
        parser.add_argument('--eval-border', type=int, default=-1, metavar='EB',
                            help='specify a border that is excluded from evaluating loss and error')
        parser.add_argument('--test-split', type=str, default='', metavar='TEST',
                            help='specify a subset for testing')
        parser.add_argument('--lowres-mode', type=str, default='', metavar='LM',
                            help='overwrite how lowres samples are generated')
        parser.add_argument('--factor', type=int, default=8, metavar='R',
                            help='upsampling factor')
        parser.add_argument('--zero-guidance', default=False, action='store_true',
                            help='use zeros for guidance')
        parser.add_argument('--val-ratio', type=float, default=0.0, metavar='V',
                            help='use this portion of training set for validation')
        parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                            help='input batch size for testing')
        parser.add_argument('--test-crop', type=int, default=128, metavar='CROP',
                            help='input crop size in training')
        parser.add_argument('--dataset_mode', type=str, default='nyu')
        parser.add_argument('--save_result', type=str, default='')
        parser.add_argument('--is_cuda', type=bool, default=True)
        parser.add_argument('--ssgnet_pretrained', type=str, default='')

        self.opts = parser.parse_args()

    @property
    def parse(self):
        opts_dict = OrderedDict(vars(self.opts))
        pprint.pprint(opts_dict)

        return self.opts
