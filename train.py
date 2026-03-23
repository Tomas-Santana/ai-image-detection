import os
from typing import Sized, cast
import torch
from tensorboardX import SummaryWriter
import numpy as np
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions
from data.dataloader import get_loader
from sklearn.metrics import average_precision_score, accuracy_score, roc_curve, auc
from tqdm import tqdm

def validate(model, data_loader):
    device = next(model.parameters()).device
    print("number of validation images: ", len(cast(Sized, data_loader.dataset)))
    with torch.no_grad():
        y_true, y_pred = [], []
        for data in data_loader:
            input_img = data[0]
            cropped_img = data[1].to(device)
            label = data[2].to(device)
            scale = data[3].to(device)

            y_pred.extend(model(input_img, cropped_img, scale).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
 
    return acc, roc_auc, ap


if __name__ == '__main__':
    opt = TrainOptions().parse_args()
    train_loader = get_loader(
        opt.train_dataset_options,
        train=True,
        input_size=opt.load_size,
        crop_size=opt.crop_size,
    )
    val_loader = get_loader(
        opt.val_dataset_options,
        train=False,
        input_size=opt.load_size,
        crop_size=opt.crop_size,
    )
    dataset_size = len(cast(Sized, train_loader.dataset))
    print('#training images = %d' % dataset_size)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.experiment_name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.experiment_name, "val"))

    model = Trainer(opt)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_params)

    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.0001, verbose=True)
    for epoch in range(opt.niter):
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.niter}"):
            model.total_steps += 1

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.checkpoint_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

            if model.total_steps % opt.checkpoint_freq == 0:
                print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                      (opt.experiment_name, epoch, model.total_steps))
                model.save_networks('latest')


        print('saving the model at the end of epoch %d, iters %d' % (epoch, model.total_steps))
        model.save_networks(epoch)

        # Validation 
        model.eval()
        acc, roc_auc, ap = validate(model.model, val_loader)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('roc_auc', roc_auc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; roc_auc: {}; ap: {}".format(epoch, acc, roc_auc, ap))
        info = [str(epoch), ',', str(acc), ',', str(roc_auc), ',', str(ap)]
        with open('./evalacc.txt', 'a') as f:
            f.writelines(info)
            f.writelines('\n')
        
        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 2, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.00005, verbose=True)
            else:
                print("Learning rate dropped to minimum, still training with minimum learning rate...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.00005, verbose=True)
                break

        model.train()
