import os
from typing import Any, Sized, cast
import torch
from tensorboardX import SummaryWriter
import numpy as np
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions
from data.dataloader import get_loader
from sklearn.metrics import average_precision_score, accuracy_score, roc_curve, auc
from storage.default import is_azure_blob_url
from tqdm import tqdm


def _resolve_tensorboard_log_root(checkpoints_dir: str, experiment_name: str) -> str:
    if is_azure_blob_url(checkpoints_dir):
        local_log_root = os.path.join('./checkpoints', experiment_name)
        print(
            "checkpoints_dir is an Azure Blob URL; TensorBoard logs will be written locally to %s"
            % local_log_root
        )
        return local_log_root
    return os.path.join(checkpoints_dir, experiment_name)


def _append_evalacc_row(epoch: int, acc: float, roc_auc: float, ap: float) -> None:
    info = [str(epoch), ',', str(acc), ',', str(roc_auc), ',', str(ap)]
    with open('./evalacc.txt', 'a', encoding='utf-8') as f:
        f.writelines(info)
        f.writelines('\n')


def _build_google_sheets_reporter(opt: TrainOptions) -> Any | None:
    if not opt.save_results_to_google_sheets:
        return None

    from reporting.google_sheets_reporter import GoogleSheetsEpochReporter

    credentials_path = opt.google_sheets_credentials_path or None
    return GoogleSheetsEpochReporter(
        spreadsheet_id=opt.google_sheets_spreadsheet_id,
        experiment_name=opt.experiment_name,
        credentials_path=credentials_path,
    )


def _append_google_sheets_row(
    reporter: Any | None,
    epoch: int,
    train_loss: float,
    acc: float,
    roc_auc: float,
    ap: float,
    total_steps: int,
) -> None:
    if reporter is None:
        return
    reporter.append_epoch_result(
        epoch=epoch,
        train_loss=train_loss,
        accuracy=acc,
        roc_auc=roc_auc,
        average_precision=ap,
        total_steps=total_steps,
    )

def validate(model, data_loader):
    device = next(model.parameters()).device
    amp = cast(Any, torch.amp)
    use_amp = device.type == 'cuda'
    try:
        print("number of validation images: ", len(cast(Sized, data_loader.dataset)))
    except TypeError:
        print("number of validation images: unknown (iterable dataset)")

    with torch.no_grad():
        y_true, y_pred = [], []
        for data in data_loader:
            input_img = data[0]
            cropped_img = data[1].to(device)
            label = data[2].to(device)
            scale = data[3].to(device)

            with amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                logits = model(input_img, cropped_img, scale)
            y_pred.extend(logits.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = float(accuracy_score(y_true, y_pred > 0.5))
    unique_labels = np.unique(y_true)

    if np.any(y_true == 1):
        ap = float(average_precision_score(y_true, y_pred))
    else:
        ap = float('nan')

    if unique_labels.size >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = float(auc(fpr, tpr))
    else:
        roc_auc = float('nan')
        print(
            f"Validation labels contain one class only ({unique_labels.tolist()}); roc_auc set to NaN for this epoch"
        )
 
    return acc, roc_auc, ap


if __name__ == '__main__':
    opt = TrainOptions().parse_args()
    amp = cast(Any, torch.amp)
    use_amp = len(opt.gpu_ids) > 0 and torch.cuda.is_available()
    scaler = amp.GradScaler("cuda", enabled=use_amp)
    google_sheets_reporter = _build_google_sheets_reporter(opt)
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
    try:
        dataset_size = len(cast(Sized, train_loader.dataset))
        print('#training images = %d' % dataset_size)
    except TypeError:
        print('#training images = unknown (iterable dataset)')

    tensorboard_log_root = _resolve_tensorboard_log_root(opt.checkpoints_dir, opt.experiment_name)
    train_writer = SummaryWriter(os.path.join(tensorboard_log_root, "train"))
    val_writer = SummaryWriter(os.path.join(tensorboard_log_root, "val"))

    model = Trainer(opt)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_params)

    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.0001, verbose=True)
    for epoch in range(opt.niter):
        epoch_loss_total = 0.0
        epoch_step_count = 0
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.niter}"):
            model.total_steps += 1

            model.set_input(data)
            model.optimizer.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                model.forward()
                loss = model.get_loss()

            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
            model.loss = float(loss.detach().item())
            epoch_loss_total += model.loss
            epoch_step_count += 1

            if model.total_steps % opt.checkpoint_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

            if model.total_steps % opt.checkpoint_freq == 0:
                print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                      (opt.experiment_name, epoch, model.total_steps))
                model.save_networks('latest')
                
        model.scheduler.step()
        epoch_train_loss = epoch_loss_total / epoch_step_count if epoch_step_count > 0 else float('nan')

        if epoch % opt.save_model_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, model.total_steps))
            model.save_networks(epoch)

        # Validation 
        model.eval()
        acc, roc_auc, ap = validate(model.model, val_loader)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('roc_auc', roc_auc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print(
            "(Val @ epoch {}) train_loss: {}; acc: {}; roc_auc: {}; ap: {}".format(
                epoch,
                epoch_train_loss,
                acc,
                roc_auc,
                ap,
            )
        )
        _append_evalacc_row(epoch, acc, roc_auc, ap)
        _append_google_sheets_row(
            reporter=google_sheets_reporter,
            epoch=epoch,
            train_loss=epoch_train_loss,
            acc=acc,
            roc_auc=roc_auc,
            ap=ap,
            total_steps=model.total_steps,
        )
        
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
