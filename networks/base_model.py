# from pix2pix
import io
import torch
import torch.nn as nn
from storage.default import get_storage_fs


class BaseModel(nn.Module):
    model: nn.Module
    optimizer: torch.optim.Optimizer
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.is_train = opt.is_train
        self.checkpoint_fs = get_storage_fs(opt.checkpoints_dir)
        self.save_dir = self.checkpoint_fs.join_path(opt.checkpoints_dir, opt.experiment_name)
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    def save_networks(self, epoch):
        save_filename = 'model_epoch_%s.pth' % epoch
        save_path = self.checkpoint_fs.join_path(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'total_steps' : self.total_steps,
        }

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        self.checkpoint_fs.write_bytes(save_path, buffer.getvalue())

    # load models from the disk
    def load_networks(self, load_path):
        # load_filename = 'model_epoch_%s.pth' % epoch
        # load_path = os.path.join(self.save_dir, load_filename)

        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        load_fs = get_storage_fs(load_path)
        checkpoint_bytes = load_fs.read_bytes(load_path)
        state_dict = torch.load(io.BytesIO(checkpoint_bytes), map_location=self.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        from collections import OrderedDict
        new_state_dict = {}
        for k,v in state_dict.items():
            if isinstance(v, OrderedDict):
                newdict = OrderedDict([(k[7:], v) if k[:7] == 'module.' else (k, v) for k, v in v.items()])
                new_state_dict[k] = newdict
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict['model'])
        self.total_steps = new_state_dict['total_steps']

        if self.is_train and not self.opt.new_optim:
            self.optimizer.load_state_dict(new_state_dict['optimizer'])
            ### move optimizer state to GPU
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            for g in self.optimizer.param_groups:
                g['lr'] = self.opt.lr

    def eval(self):
        return self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()
