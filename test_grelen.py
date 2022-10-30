
from os import path
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from lib.utils import *
from model.GRELEN import *




if __name__ == '__main__':
    from config_files.SWAT_config import Config

    config = Config()

    device = config.device
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    epochs = config.epochs
    test_filename = config.test_filename
    test_loader, test_target_tensor,_mean, _std = load_data_test(test_filename, device, batch_size*10)


    B, N, T, target_T = config.B, config.N, config.T, config.target_T
    n_in = config.n_in
    n_hid = config.n_hid
    do_prob = config.do_prob

    Graph_learner_n_hid = config.Graph_learner_n_hid
    Graph_learner_n_head_dim = config.Graph_learner_n_head_dim
    Graph_learner_head = config.Graph_learner_head
    prior = config.prior
    temperature = config.temperature
    GRU_n_dim = config.GRU_n_dim
    max_diffusion_step = config.max_diffusion_step
    num_rnn_layers = config.num_rnn_layers

    start_epoch = config.start_epoch
    params_path = config.save_dir

    num_nodes = N
    hard = 'True'
    filter_type = 'random'
    variation = 1

    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)
    log_prior = log_prior.cuda()

    net = Grelen(config.device, T, target_T, Graph_learner_n_hid, Graph_learner_n_head_dim, Graph_learner_head,
                 temperature,
                 hard, \
                 GRU_n_dim, max_diffusion_step, num_nodes, num_rnn_layers, filter_type, do_prob=0.).to(config.device)


    param_file = config.param_file

    net.load_state_dict(torch.load(param_file))

    print('Model loaded...')
    target_tensor = torch.zeros((0, N, target_T))
    reconstructed_tensor = torch.zeros((0, N, target_T))
    prob_tensor = torch.zeros((0, N * (N - 1), Graph_learner_head))

    with torch.no_grad():
        for batch_index, batch_data in enumerate(test_loader):
            encoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs[:, :, 0, :]
            labels = labels[:, :, 0, 1:]
            prob, output = net(encoder_inputs)
            prob_tensor = torch.cat([prob_tensor, prob.cpu()], dim=0)

    prob_result = prob_tensor.cpu().detach().numpy()
    mat_test = reshape_edges(prob_tensor, N).cpu()

    save_path = path.dirname(param_file)
    if config.save_result == True:
        np.save(save_path+'/'+os.path.basename(param_file).split('.')[0]+'.npy', np.array(mat_test))
    print('Graph saved...')

    ### Evaluation
    print('Evaluation...')
    w = config.moving_window_
    anomaly_time = pd.read_csv(config.anomaly_file)
    anomaly_time = np.array(anomaly_time.iloc[:, :2])
    anomaly_start = anomaly_time[:, 0]
    anomaly_end = anomaly_time[:, 1]


    total_out_degree_move_filtered = np.zeros((mat_test.shape[1] - w + 1, N))
    for fe in range(N):
        y = (torch.mean(mat_test[0, :, :, fe], -1))
        xx = moving_average(y, w)
        total_out_degree_move_filtered[:, fe] = y[w-1:] - xx

    loss = np.mean(total_out_degree_move_filtered, 1)
    f1 = np.zeros((200, 200))
    for i in range(200):
        for j in range(200):
            thr1 = 0.0005 * i
            thr2 = -0.0005 * j
            anomaly, ground_truth = point_adjust_eval(anomaly_start, anomaly_end, config.downsampling_fre, (loss), thr1, thr2)
            f1[i, j] = f1_score(anomaly, ground_truth)
    pos = np.unravel_index(np.argmax(f1), f1.shape)
    anomaly, ground_truth = point_adjust_eval(anomaly_start, anomaly_end, config.downsampling_fre, (loss), pos[0] * 0.0005,
                                                      -pos[1] * 0.0005)
    print('F1 score: ', f1_score(anomaly, ground_truth))
    print('Precision score: ', precision_score(anomaly, ground_truth))
    print('Recall score: ', recall_score(anomaly, ground_truth))
    print('Confusion matrix: ', classification_report(anomaly, ground_truth))






