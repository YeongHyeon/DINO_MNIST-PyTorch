import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class Neuralnet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.who_am_i = "DINO"

        self.dim_h = kwargs['dim_h']
        self.dim_w = kwargs['dim_w']
        self.dim_c = kwargs['dim_c']
        self.dim_out = kwargs['dim_out']
        self.k_size = kwargs['k_size']
        self.filters = kwargs['filters']

        # self.center = 1
        self.register_buffer("center", torch.zeros(1, self.dim_out))
        self.learning_rate = kwargs['learning_rate']
        self.path_ckpt = kwargs['path_ckpt']

        self.ngpu = kwargs['ngpu']
        self.device = kwargs['device']

        self.params, self.names = [], []
        self.params.append(WarmupConv(self.dim_c, self.filters[0], self.k_size, stride=1, name="warmup").to(self.device))
        self.names.append("warmupconv")

        for idx_filter, filter in enumerate(self.filters[:-1]):

            self.params.append(ConvBlock(self.filters[idx_filter], self.filters[idx_filter]*2, \
                self.k_size, stride=1, name="conv%d_1" %(idx_filter+1)).to(self.device))
            self.names.append("conv%d_1" %(idx_filter+1))
            self.params.append(ConvBlock(self.filters[idx_filter]*2, self.filters[idx_filter]*2, \
                self.k_size, stride=2, name="conv%d_2" %(idx_filter+1)).to(self.device))
            self.names.append("conv%d_2" %(idx_filter+1))

        self.params.append(Classifier(self.filters[-1], self.dim_out, name='classifier').to(self.device))
        self.names.append("classifier")
        self.modules = nn.ModuleList(self.params)

    def forward(self, x, training=False, centering=False, temperature=1):

        for idx_param, _ in enumerate(self.params):
            x = self.params[idx_param](x)

        if(training):
            if(centering): x = (x - self.center)
            y_hat = F.softmax(x/temperature, dim=1)
        else:
            y_hat = x

        return {'y_hat':y_hat}

    def update_center(self, y_hat):

        batch_center = torch.sum(y_hat, dim=0, keepdim=True)
        batch_center = batch_center / y_hat.size(0)

        self.center = self.center * 0.9 + batch_center * (1 - 0.9)

class WarmupConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, name=""):
        super().__init__()
        self.warmup = nn.Sequential()
        self.warmup.add_module("%s_conv" %(name), nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2))
        self.warmup.add_module("%s_act" %(name), nn.ReLU())

    def forward(self, x):

        out = self.warmup(x)
        return out

class Classifier(nn.Module):

    def __init__(self, in_channels, out_channels, name=""):
        super().__init__()
        self.clf = nn.Sequential()
        self.clf.add_module("%s_lin0" %(name), nn.Linear(in_channels, int(in_channels*1.5)))
        self.clf.add_module("%s_act0" %(name), nn.ReLU())
        self.clf.add_module("%s_lin1" %(name), nn.Linear(int(in_channels*1.5), out_channels))

    def forward(self, x):

        gap = torch.mean(x, axis=(2, 3))
        return self.clf(gap)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=True, name=""):
        super().__init__()

        self.out_channels = out_channels

        self.conv_ = nn.Sequential()
        self.conv_.add_module("%s_conv" %(name), nn.Conv2d(in_channels, out_channels, \
            kernel_size, stride, padding=kernel_size//2)) 
        self.conv_.add_module("%s_act" %(name), nn.ReLU())

    def forward(self, x):
        return self.conv_(x)
