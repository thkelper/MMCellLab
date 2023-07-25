from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from mmcv.runner import HOOKS, LrUpdaterHook


@HOOKS.register_module()
class CosineAnnealingWarmRestartsLrUpdaterHook(LrUpdaterHook):
    def __init__(self,
                 T_0=1,
                 eta_min=1e-7,
                 by_epoch=False,
                 **kwargs):
        super(CosineAnnealingWarmRestartsLrUpdaterHook, self).__init__(by_epoch, **kwargs)
        self.updater = CosineAnnealingWarmRestarts(T_0, eta_min=eta_min)

    def get_lr(self, runner, base_lr):
        return self.updater.get_lr()
