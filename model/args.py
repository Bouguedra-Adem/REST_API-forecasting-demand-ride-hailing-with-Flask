import pandas as pd
class Args:
    def __init__(self):
        self.cuda = True
        self.no_cuda = False
        self.seed = 1
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 10
        self.lr = 0.01
        self.momentum = 0.5
        self.log_interval = 10
        self.img_size=9
        self.num_filtre=64
        self.size_filtre=5
        self.kernel_maxpooling=2
        self.stride_maxpooling=2
        self.output_size_linear=64
        self.hidden_size=16
        self.output_size_linear_lstm=1
        self.batsh_size=15
        self.seq_len=8
        #self.date_rng = pd.to_datetime(pd.date_range(start='2018-01-01 00:00:00', end='2019-12-30 00:00:00', freq='H'))
        self.number_of_zone_training=15

