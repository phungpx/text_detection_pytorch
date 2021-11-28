import argparse
import collections
import torch
import numpy as np

from model import loss as module_loss
from model import model as module_arch
from model import metric as module_metric
from dataloader import mnist_dataset as module_data

from trainer import Trainer
from utils import ConfigParser
from utils import prepare_device


# Fix random seeds for reproducibility.
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config):
    logger = config.get_logger('train')

    # Setup data_loader instances.
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # Build model architecture, then print to console.
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # Prepare for (multi-device) GPU training.
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, metric) for metric in config['metrics']]

    # Build optimizer.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    # Build learning rate scheduler.
    # Note: delete every lines containing lr_scheduler for disabling scheduler.
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        criterion=criterion,
        metric_ftns=metrics,
        optimizer=optimizer,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler
    )

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument(
        '-c', '--config', default=None, type=str,
        help='config file path (default: None)'
    )
    args.add_argument(
        '-r', '--resume', default=None, type=str,
        help='path to latest checkpoint (default: None)'
    )
    args.add_argument(
        '-d', '--device', default=None, type=str,
        help='indices of GPUs to enable (default: all)'
    )

    # Custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
