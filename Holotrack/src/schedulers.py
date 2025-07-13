import torch
import math

class CustomPolynomialLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_epochs, end_lr=1e-6, power=1.0, last_epoch=-1):
        self.max_epochs = max_epochs
        self.end_lr = float(end_lr)
        self.power = float(power)
        super(CustomPolynomialLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        epoch = self.last_epoch
        decay = (1 - (epoch / float(self.max_epochs))) ** self.power
        return [(base_lr - self.end_lr) * decay + self.end_lr for base_lr in self.base_lrs]


class CosineAnnealingWithRestartsLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, t_max, eta_min=1e-6, factor=2.0, last_epoch=-1):
        self.t_max = t_max
        self.eta_min = eta_min
        self.factor = factor
        self.current_cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWithRestartsLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur >= self.t_max:
            self.current_cycle += 1
            self.T_cur = 0
            self.t_max = int(self.t_max * self.factor)
        
        cos_decay = 0.5 * (1 + math.cos(math.pi * self.T_cur / self.t_max))
        return [self.eta_min + (base_lr - self.eta_min) * cos_decay for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            self.T_cur += 1
        else:
            if epoch >= self.last_epoch:
                self.T_cur = epoch - sum(int(self.t_max * self.factor ** i) for i in range(self.current_cycle))
        super().step(epoch)


class OneCycleCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, t_max, max_lr=0.01, eta_min=1e-6, last_epoch=-1):
        self.t_max = t_max
        self.max_lr = max_lr
        self.eta_min = eta_min
        super(OneCycleCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        cos_decay = 0.5 * (1 + math.cos(math.pi * self.last_epoch / self.t_max))
        return [self.eta_min + (self.max_lr - self.eta_min) * cos_decay for _ in self.base_lrs]
