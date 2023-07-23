import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import source.utils as utils

import torchsummary
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

class Agent(object):

    def __init__(self, **kwargs):

        print("\nInitializing Neural Network...")
        self.nn = kwargs['nn']
        self.dim_h = kwargs['dim_h']
        self.dim_w = kwargs['dim_w']
        self.dim_c = kwargs['dim_c']
        self.dim_out = kwargs['dim_out']
        self.k_size = kwargs['k_size']
        self.filters = kwargs['filters']
        tmp_filters = self.filters.split(',')
        for idx, _ in enumerate(tmp_filters):
            tmp_filters[idx] = int(tmp_filters[idx])
        self.filters = tmp_filters

        self.learning_rate = kwargs['learning_rate']
        self.path_ckpt = kwargs['path_ckpt']

        self.ngpu = kwargs['ngpu']
        self.device = kwargs['device']

        self.__student = self.nn.Neuralnet(dim_h=self.dim_h, dim_w=self.dim_w, dim_c=self.dim_c, dim_out=self.dim_out, \
            k_size=self.k_size, filters=self.filters, \
            learning_rate=self.learning_rate, path_ckpt=self.path_ckpt, \
            ngpu=self.ngpu, device=self.device).to(self.device)
        self.__teacher = self.nn.Neuralnet(dim_h=self.dim_h, dim_w=self.dim_w, dim_c=self.dim_c, dim_out=self.dim_out, \
            k_size=self.k_size, filters=self.filters, \
            learning_rate=self.learning_rate, path_ckpt=self.path_ckpt, \
            ngpu=self.ngpu, device=self.device).to(self.device)

        self.__init_propagation(path=self.path_ckpt)

    def __init_propagation(self, path):

        utils.make_dir(self.path_ckpt, refresh=False)
        self.save_params()

        out = torchsummary.summary(self.__student, (self.dim_c, self.dim_h, self.dim_w))

        self.optimizer = optim.Adam(self.__student.parameters(), lr=self.learning_rate)
        self.writer = SummaryWriter(log_dir=self.path_ckpt)

    def step(self, minibatch, iteration=0, training=False):

        x1, x2, y = minibatch['x1'], minibatch['x2'], minibatch['y']
        x1, x2, y = torch.tensor(utils.nhwc2nchw(x1)), torch.tensor(utils.nhwc2nchw(x2)), torch.tensor(y)
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        y = y.to(self.device)

        if(training):
            self.optimizer.zero_grad()

        step_dict_s = self.__student(x1, training=training, centering=False, temperature=1)
        with torch.no_grad():
            step_dict_t = self.__teacher(x2, training=training, centering=True, temperature=0.5)

        y_s = step_dict_s['y_hat']
        y_t = step_dict_t['y_hat']

        losses = self.loss(y_s, y_t)

        if(training):
            losses['opt'].backward()
            self.optimizer.step()

        if(training):
            self.writer.add_scalar("%s/opt" %("DINO"), scalar_value=losses['opt'], global_step=iteration)
            self.writer.add_scalar("%s/lr" %("DINO"), scalar_value=self.optimizer.param_groups[0]['lr'], global_step=iteration)

        self.EMA()
        self.__teacher.update_center(y_t)
        
        with torch.no_grad():
            step_dict_s = self.__student(x1, training=False, centering=False, temperature=1)
            step_dict_t = self.__teacher(x2, training=False, centering=True, temperature=0.5)

        y_s = step_dict_s['y_hat']
        y_t = step_dict_t['y_hat']
        
        for key in list(losses.keys()):
            losses[key] = self.detach(losses[key])

        return {'y_s':self.detach(y_s), 'y_t':self.detach(y_t), 'losses':losses}

    def EMA(self, lamb=0.996):

        with torch.no_grad():
            for idx_param in range(len(self.__student.params)):
                list_key_s = self.__student.params[idx_param].state_dict()
                tmp_dict = {}
                for idx_key, name_key in enumerate(list_key_s):
                    tmp_dict[name_key] = \
                        (self.__teacher.params[idx_param].state_dict()[name_key]*lamb) \
                        + (self.__student.params[idx_param].state_dict()[name_key] * (1-lamb))
                
                self.__teacher.params[idx_param].load_state_dict(tmp_dict)

    def loss(self, y_s, y_t):

        loss_ce = nn.CrossEntropyLoss(reduce=False)
        opt_b = loss_ce(y_s, target=y_t)
        opt = torch.mean(opt_b)

        return {'opt_b': opt_b, 'opt': opt}

    def save_params(self, model='base'):

        torch.save(self.__student.state_dict(), os.path.join(self.path_ckpt, '%s.pth' %(model)))

    def load_params(self, model):

        self.__student.load_state_dict(torch.load(os.path.join(self.path_ckpt, '%s' %(model))), strict=False)
        self.__student.eval()

    def detach(self, x):

        try: x = x.detach().numpy()
        except: x = x.cpu().detach().numpy()

        return x
