import torch.optim as optim

def get_optimizer(net, config):
    optimizer = None
    if config.optimizer_name == 'SGD':
        # nesterov=True, weight_decay = 0.0005
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer_name == 'Adam':
        # betas=(0.9, 0.999), weight_decay=0.05
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), eps=config.eps, lr=config.lr, betas=config.betas, weight_decay=config.weight_decay)

    return optimizer
if __name__ == '__main__':
    net = None
    config = None
    optimizer = get_optimizer(net, config)