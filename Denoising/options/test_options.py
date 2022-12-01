import argparse
import pprint
from collections import OrderedDict

class TestOptions:
    def __init__(self):
        parser = argparse.ArgumentParser()

        # ------------------
        # dataset parameters
        # ------------------
        parser.add_argument('--save_dir', type=str, default='')
        parser.add_argument('--ssg-pretrained', type=str, default='')
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--print_freq', type=int, default=1)
        parser.add_argument('--validation', type=int, default=1)
        parser.add_argument('--is_cuda', type=bool, default=True)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--noise_type', type=str, default='g')
        parser.add_argument('--noise_level', type=list, default=50)
        parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                            help='input batch size for testing')

        self.opts = parser.parse_args()

    @property
    def parse(self):
        opts_dict = OrderedDict(vars(self.opts))
        pprint.pprint(opts_dict)

        return self.opts
