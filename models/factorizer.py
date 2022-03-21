import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ExponentialLR
from copy import deepcopy

from models.modules import LR, FM, DeepFM, AutoInt
from utils.train import use_cuda, use_optimizer, get_grad_norm


def setup_factorizer(opt):
    new_opt = deepcopy(opt)
    for k, v in opt.items():
        if k.startswith('fm_'):
            new_opt[k[3:]] = v
    return FMFactorizer(new_opt)

class Factorizer(object):
    '''
    这是个什么东西呢？@todo

    '''
    def __init__(self, opt):
        self.opt = opt
        self.clip = opt.get('grad_clip')
        self.use_cuda = opt.get('use_cuda')
        self.batch_size_test = opt.get('batch_size_test')
        self.l2_penalty = opt['l2_penalty']

        self.criterion = BCEWithLogitsLoss(size_average=False)

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.param_grad = None
        self.optim_status = None

        self.prev_param = None
        self.param = None

        self._train_step_idx = None
        self._train_episode_idx = None

    @property
    def train_step_idx(self):
        return self._train_step_idx

    @train_step_idx.setter
    def train_step_idx(self, new_step_idx):
        self._train_step_idx = new_step_idx

    @property
    def train_episode_idx(self):
        return self._train_episode_idx

    @train_episode_idx.setter
    def train_episode_idx(self, new_episode_idx):
        self._train_episode_idx = new_episode_idx

    def get_grad_norm(self):
        assert hasattr(self, 'model')
        return get_grad_norm(self.model)

    def get_emb_dims(self):
        return self.model.get_emb_dims()

    def update(self, sampler):
        '''
        做一些设置。包括重置。没有真正的训练。在子类里，是真正的训练。
        '''
        if (self.train_step_idx > 0) and (self.train_step_idx % sampler.num_batches_train == 0):
            self.scheduler.step()

        self.train_step_idx += 1

        # 设置model为training mode
        self.model.train()
        self.optimizer.zero_grad()


class FMFactorizer(Factorizer):
    '''
    FM模型相关的
    '''
    def __init__(self, opt):
        super(FMFactorizer, self).__init__(opt)
        self.opt = opt
        if opt['model'] == 'linear':
            self.model = LR(opt)
        elif opt['model'] == 'fm':
            self.model = FM(opt)
        elif opt['model'] == 'deepfm':
            self.model = DeepFM(opt)
        elif opt['model'] == 'autoint':
            self.model = AutoInt(opt)
        else:
            raise ValueError("Invalid FM model type: {}".format(opt['model']))

        if self.use_cuda:
            use_cuda(True, opt['device_id'])
            self.model.cuda()

        # todo
        self.optimizer = use_optimizer(self.model, opt)
        self.scheduler = ExponentialLR(self.optimizer, gamma=opt['lr_exp_decay'])

    def init_episode(self):
        opt = self.opt
        if opt['model'] == 'linear':
            self.model = LR(opt)
        elif opt['model'] == 'fm':
            self.model = FM(opt)
        elif opt['model'] == 'deepfm':
            self.model = DeepFM(opt)
        elif opt['model'] == 'autoint':
            self.model = AutoInt(opt)
        else:
            raise ValueError("Invalid FM model type: {}".format(opt['model']))

        self._train_step_idx = 0
        if self.use_cuda:
            use_cuda(True, opt['device_id'])
            self.model.cuda()
        self.optimizer = use_optimizer(self.model, opt)
        self.scheduler = ExponentialLR(self.optimizer, gamma=opt['lr_exp_decay'])

    def update(self, sampler):
        """
        训练！！！

        update FM model parameters
        """
        # 调用父类的方法，做一下简单的设置
        super(FMFactorizer, self).update(sampler)

        data, labels = sampler.get_sample('train')
        if self.use_cuda:
            data, labels = data.cuda(), labels.cuda()

        prob_preference = self.model.forward(data)

        # 计算loss
        non_reg_loss = self.criterion(prob_preference, labels.float()) / (data.size()[0])

        # l2，且均值一下
        l2_loss = self.model.l2_penalty(data, self.l2_penalty) / (data.size()[0])
        loss = non_reg_loss + l2_loss

        # 计算叶子节点的梯度
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        # parameter update
        self.optimizer.step()

        return loss.item()
