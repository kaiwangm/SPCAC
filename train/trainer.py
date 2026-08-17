import os
import random
import yaml
import torch
import numpy as np
import wandb
import dataset
import model.zoo as zoo
from model.core.rate_distortion import rate_distortion_loss
from torch import optim
from rich.progress import track


def create_trainer(cfg_name: str, model_name: str, quality: int, dataset: str = None):
    """ Create a trainer based on the configuration """
    print('----- Creating trainer -----')
    print(f'Config: {cfg_name}')
    print(f'Model: {model_name}')
    print(f'Quality: {quality}')
    print('----------------------------')
    model_trainer = trainer()
    model_trainer.initialize(cfg_name, model_name, quality, dataset)
    return model_trainer


class trainer:
    """ Trainer class """

    def __init__(self):
        self.cfg = None
        self.train_dataset = None
        self.train_dataloader = None
        self.model = None
        self.optimizer = None
        self.aux_optimizer = None
        self.scheduler = None
        self.criterion = None
        self.now_epoch = None
        self.default_save_dir = None

    def initialize(self, cfg_name: str, model_name: str, quality: int, dataset: str = None):
        print('Initializing trainer...')
        config_dir = 'configs/trainer'

        cfg_path = os.path.join(config_dir, f'{cfg_name}.yaml')
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f'Config file not found: {cfg_path}')

        with open(cfg_path, encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        # Dataset: CLI argument takes priority over the config default
        if dataset is not None:
            self.cfg['dataset'] = dataset

        # ------------------- random seed -------------------
        random.seed(self.cfg['seed'])
        np.random.seed(self.cfg['seed'])
        torch.manual_seed(self.cfg['seed'])
        torch.cuda.manual_seed_all(self.cfg['seed'])

        # -------------------- dataset -------------------
        self.train_dataset = dataset.load_dataset(
            self.cfg['dataset'],
            mode='train'
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=self.cfg['shuffle'],
            num_workers=self.cfg['num_workers']
        )

        # -------------------- model -------------------
        if torch.cuda.is_available() is False:
            raise RuntimeError('CUDA is not available.')
        torch.backends.cudnn.benchmark = True

        self.model, lam = zoo.load_model(
            model_name,
            quality=quality,
            category=self.train_dataset.category
        )

        self.model.cuda()
        self.model.train()

        self.default_save_dir = os.path.join(
            self.cfg['checkpoint_save_dir'],
            self.cfg['dataset'],
            self.model.__class__.__name__,
            f'quality_{quality}'
        )

        # -------------------- optimizer -------------------
        parameters_net = {
            param
            for name, param in self.model.named_parameters()
            if param.requires_grad and not name.endswith(".quantiles")
        }

        parameters_aux = {
            param
            for name, param in self.model.named_parameters()
            if param.requires_grad and name.endswith(".quantiles")
        }

        parameters = {
            "net": parameters_net,
            "aux": parameters_aux,
        }

        self.optimizer = optim.Adam(
            parameters["net"],
            lr=self.cfg['lr_net_init']
        )
        self.aux_optimizer = optim.Adam(
            parameters["aux"],
            lr=self.cfg['lr_aux_init']
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.cfg['lr_milestones'],
            gamma=self.cfg['lr_decay']
        )

        self.criterion = rate_distortion_loss(
            lam=lam,
            attributes=self.train_dataset.attributes,
            metric='mse'
        )

    def train(self):
        if self.cfg['use_wandb']:
            wandb.init(project=self.cfg['wandb_project'], config=self.cfg)

        self.model.train()
        for epoch in range(self.cfg['start_epoch'], self.cfg['end_epoch']):
            self.now_epoch = epoch
            self.train_one_epoch()
            self.validate_one_epoch()

            # Save the latest checkpoint after each epoch
            self.save(
                os.path.join(
                    self.default_save_dir,
                    'eb_las.pth'
                )
            )

            # Save intermediate checkpoints periodically
            if self.now_epoch % self.cfg['save_epoch'] == 0:
                self.save(
                    os.path.join(
                        self.default_save_dir,
                        f'eb_e{self.now_epoch}.pth'
                    )
                )

    def save(self, save_path: str):
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.model.state_dict(), save_path)

    def train_one_epoch(self):
        for _, data in track(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            description=f'Training Epoch {self.now_epoch}'
        ):
            points, colors = data
            points = points.int().cuda()
            colors = colors.float().cuda()

            self.optimizer.zero_grad()
            self.aux_optimizer.zero_grad()

            # ---------------------- forward ----------------------
            out_net = self.model(points, colors)
            out_criterion = self.criterion(out_net)
            train_loss = out_criterion['loss']

            # ---------------------- net loss ----------------------
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # ---------------------- aux loss ----------------------
            aux_loss = self.model.aux_loss()
            aux_loss.backward()
            self.aux_optimizer.step()

            if self.cfg['use_wandb']:
                wandb.log(
                    {
                        'loss': train_loss,
                        'bpp': out_criterion['loss_R'],
                        'mse': out_criterion['loss_D'],
                        'psnr': out_criterion['psnr'],
                        'loss_R_y': out_criterion['loss_R_y'],
                        'loss_R_z': out_criterion['loss_R_z'] if 'loss_R_z' in out_criterion else 0.0,
                        'aux_loss': aux_loss,
                        'epoch': self.now_epoch,
                        'lr': self.optimizer.param_groups[0]['lr']
                    }
                )

            torch.cuda.empty_cache()

        self.scheduler.step()

    def validate_one_epoch(self):
        return
