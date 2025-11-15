import sys
sys.path.append('.')
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse, os, json, time, datetime, yaml
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, average_precision_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
from torchmetrics.classification import MultilabelAveragePrecision, AveragePrecision
from torchmetrics.functional.classification.average_precision import average_precision

from dataset.datasets import MultiTaskDataset
from model.mlp import MultiTaskMLP
from utils import commons

import warnings
# suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
torch.set_num_threads(4)


def train(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, config, device, logger):
    model.train()
    best_val_loss = float('inf')
    best_val_aupr_mean = 0.0
    n_bad = 0
    epsilon = 1e-4
    for epoch in range(config.train.num_epochs):
        epoch_start = time.time()
        train_losses = []
        train_losses_per_task = [[] for _ in range(len(config.data.task_list))]
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.train.num_epochs}'):
            embeddings, labels_per_task = batch
            embeddings = embeddings.to(device)
            labels_per_task = [labels.to(device) for labels in labels_per_task]

            optimizer.zero_grad()
            outputs_per_task = model(embeddings)

            loss = 0
            for i, (output, labels) in enumerate(zip(outputs_per_task, labels_per_task)):
                task_loss = criterion(output, labels)
                loss += task_loss
                train_losses_per_task[i].append(task_loss.item())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)
        avg_train_losses_per_task = [np.mean(task_losses) for task_losses in train_losses_per_task]
        val_loss, val_loss_per_task, val_aupr_per_task = evaluate(model, val_loader, criterion, device)
        val_aupr_mean = np.mean(val_aupr_per_task)
        lr_scheduler.step(val_loss)

        logger.info(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUPR (mean): {val_aupr_mean:.4f}, Time: {time.time() - epoch_start:.2f}s')

        wandb.log({
            'Train Loss': avg_train_loss,
            'Val Loss': val_loss,
            'Val AUPR Mean': val_aupr_mean,
            **{f'Train Loss Task {i+1} ({config.data.task_list[i]})': avg_train_losses_per_task[i] for i in range(len(config.data.task_list))},
            **{f'Val Loss Task {i+1} ({config.data.task_list[i]})': val_loss_per_task[i] for i in range(len(config.data.task_list))},
            **{f'Val AUPR Task {i+1} ({config.data.task_list[i]})': val_aupr_per_task[i] for i in range(len(config.data.task_list))},
        })

        if val_aupr_mean >= best_val_aupr_mean + epsilon:
            logger.info(f'New best performance found! Val AUPR (mean): {val_aupr_mean:.4f}')
            best_val_loss = val_loss
            best_val_aupr_mean = val_aupr_mean
            n_bad = 0
            torch.save(model.state_dict(), os.path.join(config.train.ckpt_dir, 'best_checkpoint.pt'))
        else:
            n_bad += 1

        if n_bad >= config.train.patience:
            logger.info('Early stopping triggered.')
            break
    logger.info(f'Best Val AUPR (mean): {best_val_aupr_mean:.4f}')
    for i, task in enumerate(config.data.task_list):
        logger.info(f'Val AUPR Task {i+1} ({task}): {val_aupr_per_task[i]:.4f}')


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_losses = []
    val_losses_per_task = [[] for _ in range(len(val_loader.dataset.task_list))]
    all_probs_per_task = [[] for _ in range(len(val_loader.dataset.task_list))]
    all_labels_per_task = [[] for _ in range(len(val_loader.dataset.task_list))]

    with torch.no_grad():
        for batch in val_loader:
            embeddings, labels_per_task = batch
            embeddings = embeddings.to(device)
            labels_per_task = [labels.to(device) for labels in labels_per_task]

            outputs_per_task = model(embeddings)
            loss = 0
            for i, (output, labels) in enumerate(zip(outputs_per_task, labels_per_task)):
                task_loss = criterion(output, labels)
                loss += task_loss
                val_losses_per_task[i].append(task_loss.item())
            val_losses.append(loss.item())

            # Collect probabilities and labels
            probs = [torch.sigmoid(output) for output in outputs_per_task]
            for i, prob in enumerate(probs):
                all_probs_per_task[i].append(prob)
            for i, label in enumerate(labels_per_task):
                all_labels_per_task[i].append(label)

    val_loss = np.mean(val_losses).item()
    val_loss_per_task = [np.mean(task_losses).item() for task_losses in val_losses_per_task]
    all_probs_per_task = [torch.cat(probs) for probs in all_probs_per_task]
    all_labels_per_task = [torch.cat(labels) for labels in all_labels_per_task]

    # Compute AUPR
    aupr_per_task = []
    for probs, labels in zip(all_probs_per_task, all_labels_per_task):
        aupr = average_precision(probs, labels.int(), average='micro', task='binary').item()
        aupr_per_task.append(aupr)

    model.train()
    return val_loss, val_loss_per_task, aupr_per_task

def merge_config_args(config, args):
    config.train.seed = args.seed if args.seed is not None else config.train.seed
    config.train.loss = args.loss if args.loss is not None else config.train.loss

def get_args():
    parser = argparse.ArgumentParser(description='Train MLP model')
    parser.add_argument('config', type=str, default='configs/train.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='logs/mlp')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--no_timestamp', action='store_true')
    parser.add_argument('--loss', type=str, default=None, help='Loss function to use')

    args = parser.parse_args()
    return args

def main():
    start_overall = time.time()
    args = get_args()
    config = commons.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    merge_config_args(config, args)
    commons.seed_all(config.train.seed)
    log_dir = commons.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag, timestamp=not args.no_timestamp)

    wandb.init(project='train_multitask', config=vars(config), name=os.path.basename(log_dir))
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    config.train.ckpt_dir = ckpt_dir

    logger = commons.get_logger('train_mlp', log_dir)
    logger.info(args)
    logger.info(config)

    logger.info('Randomly split train and validation set')
    all_data = torch.load(config.data.train_data_file)
    with open(config.data.label_file, 'r') as f:
        label_list = json.load(f)
    pids = list(all_data.keys())
    num_train_val = len(pids)
    indices = commons.get_random_indices(num_train_val, seed=config.train.seed)
    train_indices = indices[:int(num_train_val * 0.9)]
    val_indices = indices[int(num_train_val * 0.9):]
    train_pids = [pids[i] for i in train_indices]
    val_pids = [pids[i] for i in val_indices]
    train_data = {pid: all_data[pid] for pid in train_pids}
    val_data = {pid: all_data[pid] for pid in val_pids}
    with open(os.path.join(log_dir, 'train_val_pids.json'), 'w') as f:
        json.dump({'train': train_pids, 'val': val_pids}, f)

    train_dataset = MultiTaskDataset(train_data, label_list, config.data.task_list)
    val_dataset = MultiTaskDataset(val_data, label_list, config.data.task_list)
    logger.info(f'Trainset size: {len(train_dataset)}, Validation set size: {len(val_dataset)}')
    logger.info(f'Number of labels: {len(label_list)}')
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False)

    config.model.task_out_dims = [len(label_list[task]) for task in config.data.task_list]
    model = MultiTaskMLP(
        input_size=config.model.input_dim,
        shared_hidden_sizes=config.model.shared_hidden_dims,
        task_hidden_sizes=config.model.task_hidden_dims,
        task_output_sizes=config.model.task_out_dims,
        dropout=config.model.dropout
    )
    model = model.to(args.device)
    model.train()
    logger.info(f'Model:\n{model}')
    logger.info(f'Trainable parameters: {commons.count_parameters(model)}')
    logger.info(f'Number of tasks: {len(config.data.task_list)}')

    criterion = globals()[config.train.loss]()
    optimizer = Adam(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.train.patience-10)

    commons.save_config(config, os.path.join(log_dir, 'config.yml'))

    train(model=model,
          train_loader=train_loader,
          val_loader=val_loader,
          criterion=criterion,
          optimizer=optimizer,
          lr_scheduler=lr_scheduler,
          config=config,
          device=args.device,
          logger=logger)

if __name__ == '__main__':
    main()