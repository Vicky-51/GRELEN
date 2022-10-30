from lib.utils import *

from time import time
from tensorboardX import SummaryWriter
from model.GRELEN import *





def val_epoch(net, val_loader, sw, epoch, config):

    B, N, T, target_T = config.B, config.N, config.T, config.target_T
    prior = config.prior
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)
    log_prior = log_prior.cuda()
    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():
        tmp = []
        for batch_index, batch_data in enumerate(val_loader):

            encoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs[:, :, 0, :]
            labels = labels[:, :, 0, T-target_T:]
            prob, output = net(encoder_inputs)
            loss_kl = kl_categorical(torch.mean(prob, 1), log_prior, 1).to(device)
            loss_nll = nll_gaussian(output, labels, variation).to(device)
            loss = loss_kl + loss_nll
            tmp.append(loss.item())

        validation_loss = sum(tmp) / len(tmp)
        print('epoch: %s, validation loss: %.2f' % (epoch, validation_loss))
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss


if __name__ == '__main__':
    from config_files.SWAT_config import Config
    config = Config()

    device = config.device
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    epochs = config.epochs
    train_filename = config.train_filename

    train_loader, train_target_tensor, val_loader, val_target_tensor,  _mean, _std = load_data_train(train_filename, device, batch_size)

    B, N, T, target_T = config.B, config.N, config.T, config.target_T

    n_in = config.n_in
    n_hid = config.n_hid
    do_prob = config.do_prob

    Graph_learner_n_hid = config.Graph_learner_n_hid
    Graph_learner_n_head_dim = config.Graph_learner_n_head_dim
    Graph_learner_head =  config.Graph_learner_head
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



    net = Grelen(config.device, T, target_T, Graph_learner_n_hid, Graph_learner_n_head_dim, Graph_learner_head, temperature,
                       hard, \
                       GRU_n_dim, max_diffusion_step, num_nodes, num_rnn_layers, filter_type, do_prob=0.).to(config.device)

    ###  Training Process
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    print(net)


    best_epoch = 0
    global_step = 0
    best_val_loss = np.inf
    start_time = time()

    if start_epoch > 0:
        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)
        net.load_state_dict(torch.load(params_filename))
        print('start epoch:', start_epoch)
        print('load weight from: ', params_filename)

    # train model
    for epoch in range(start_epoch, 300):
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
        val_loss = val_epoch(net, val_loader, sw, epoch, config)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename)

        net.train()  # ensure dropout layers are in train mode
        kl_train = []
        nll_train = []
        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs[:, :, 0, :]
            labels = labels[:, :, 0, T - target_T:]

            optimizer.zero_grad()
            prob, output = net(encoder_inputs)

            loss_kl = kl_categorical(torch.mean(prob, 1), log_prior, 1).to(device)
            loss_nll = nll_gaussian(output, labels, variation).to(device)
            loss = loss_kl + loss_nll
            nll_train.append(loss_nll)
            kl_train.append(loss_kl)
            loss.to(device)
            loss.backward()
            optimizer.step()

            training_loss = loss.item()
            global_step += 1
            sw.add_scalar('training_loss', training_loss, global_step)
            sw.add_scalar('kl_loss', loss_kl.item(), global_step)
            sw.add_scalar('nll_loss', loss_nll.item(), global_step)



        nll_train_ = torch.tensor(nll_train)
        kl_train_ = torch.tensor(kl_train)
        print('epoch: %s, kl loss: %.2f, nll loss: %.2f' % (epoch,  kl_train_.mean(), nll_train_.mean()))
        if epoch == 30:
            adjust_learning_rate(optimizer, 0.0002)
        if epoch == 100:
            adjust_learning_rate(optimizer, 0.0001)

