import numpy as np
class Config(object):
    def __init__(self):
        # data configs
        self.downsampling_fre = 60
        self.target_len = 30

        # model configs
        self.device = 'cuda:0'
        self.batch_size = 128
        self.learning_rate = 0.001
        self.epochs = 200
        self.train_filename = 'data/SWAT/train_swat.npz'
        self.test_filename = 'data/SWAT/test_swat.npz'



        self.B, self.N, self.T, self.target_T = 32, 51, 30, 29
        self.n_in = 12
        self.n_hid = 64
        self.do_prob = 0.

        self.Graph_learner_n_hid = 64
        self.Graph_learner_n_head_dim = 32
        self.Graph_learner_head = 4
        self.prior = np.array([0.91, 0.03, 0.03, 0.03])
        self.temperature = 0.5
        self.GRU_n_dim = 64
        self.max_diffusion_step = 2
        self.num_rnn_layers = 1

        self.save_dir = 'experiments/swat_test'
        self.start_epoch = 0
        self.param_file = "experiments/swat_test/test_model.params"
        self.save_result = True
        self.moving_window_ = 2
        self.anomaly_file = 'data/SWAT/SWAT_Time.csv'


