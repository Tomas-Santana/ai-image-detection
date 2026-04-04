import torch
import torch.nn as nn
from networks.base_model import BaseModel
from networks.patch_model import Patch5Model
from options.train_options import TrainOptions


class Trainer(BaseModel):
    def name(self):
        return "Trainer"

    def __init__(self, opt: TrainOptions):
        super(Trainer, self).__init__(opt)

        if self.is_train and not opt.continue_train:
            self.model = Patch5Model()
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)

        if opt.continue_train:
            self.model = Patch5Model()

        if self.is_train:
            self.loss_fn = nn.BCEWithLogitsLoss()

            # Acceso compatible con DataParallel
            raw_model = getattr(self.model, 'module', self.model)

            train_stage = getattr(opt, 'train_stage', 1)
            
            stage_lr = opt.lr

            if train_stage == 1:
                # Stage 1: solo DFGM — redirige CLIP sin tocar el clasificador
                trainable_params = list(raw_model.clip_global.dfgm_modules.parameters())
                stage_lr = 5e-5
                print(f"[Trainer] Stage 1 — entrenando solo DFGM "
                      f"({sum(p.numel() for p in trainable_params):,} params, lr={stage_lr})")
            else:
                # Stage 2: FuseFormer + fusión + clasificador (DFGM ya entrenado y frozen)
                dfgm_params = list(raw_model.clip_global.dfgm_modules.parameters())
                for p in dfgm_params:
                    p.requires_grad = False

                trainable_params = (
                    list(raw_model.fusion.parameters())     +
                    list(raw_model.mha_list.parameters())   +
                    list(raw_model.fc1_local.parameters())  +
                    list(raw_model.fc.parameters())         +
                    [raw_model.cls_token, raw_model.seq_pos_embed]
                )

                # fc1_global solo existe en el modelo sin FuseFormer
                if hasattr(raw_model, 'fc1_global'):
                    trainable_params += list(raw_model.fc1_global.parameters())

                # fuse_former solo existe en el modelo con FuseFormer
                if hasattr(raw_model, 'fuse_former'):
                    trainable_params += list(raw_model.fuse_former.parameters())
                print(f"[Trainer] Stage 2 — entrenando FuseFormer+clasificador "
                      f"({sum(p.numel() for p in trainable_params):,} params")

            if opt.optim == "adam":
                self.optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=stage_lr,
                    betas=(opt.beta1, 0.999),
                    weight_decay=opt.weight_decay,
                )
            elif opt.optim == "sgd":
                self.optimizer = torch.optim.SGD(
                    trainable_params, lr=stage_lr, momentum=0.0, weight_decay=0
                )
            else:
                raise ValueError("optim should be [adam, sgd]")

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=opt.niter, eta_min=1e-6
        )

        if self.is_train and opt.continue_train:
            print(opt.model_path)
            self.load_networks(opt.model_path)
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)

        if len(opt.gpu_ids) == 0:
            self.model.to("cpu")
        else:
            self.model.to(opt.gpu_ids[0])

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 2.0
            if param_group["lr"] < min_lr:
                param_group["lr"] = min_lr
                return False
        return True

    def set_input(self, data):
        self.input_img  = data[0]
        self.cropped_img = data[1].to(self.device)
        self.label       = data[2].to(self.device).float()
        self.scale       = data[3].to(self.device).float()

    def forward(self):
        self.output = self.model(self.input_img, self.cropped_img, self.scale)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()