from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--save2disk', action='store_true', help='save prediction to disk')
        parser.add_argument('--log2tensorboard', action='store_true', help='log prediction to tensorboard')
        return parser