import torch

from torchgan.logging.logger import Logger
from torchgan.losses.loss import DiscriminatorLoss, GeneratorLoss
from torchgan.models.model import Discriminator, Generator
from torchgan.trainer.base_trainer import BaseTrainer

__all__ = ["ParallelTrainer"]


class DistributedParallelTrainer(BaseTrainer):

    def __init__(
        self,
        models,
        losses_list,
        devices,
        metrics_list=None,
        ncritic=1,
        epochs=5,
        sample_size=8,
        checkpoints="./model/gan",
        retain_checkpoints=5,
        recon="./images",
        log_dir=None,
        test_noise=None,
        nrow=8,
        **kwargs
    ):
        super(DistributedParallelTrainer, self).__init__(
            losses_list,
            metrics_list=metrics_list,
            device=None,
            ncritic=ncritic,
            epochs=epochs,
            sample_size=sample_size,
            checkpoints=checkpoints,
            retain_checkpoints=retain_checkpoints,
            recon=recon,
            log_dir=log_dir,
            test_noise=test_noise,
            nrow=nrow,
            **kwargs
        )
        self.model_names = []
        self.optimizer_names = []
        self.schedulers = []
        for key, model in models.items():
            self.model_names.append(key)
            if "args" in model:
                setattr(
                    self, key, (model["name"](**model["args"])).cuda()
                )
            else:
                setattr(self, key, (model["name"]()).cuda())
            for m in getattr(self, key)._modules:
                getattr(self, key)._modules[m] = torch.nn.parallel.DistributedDataParallel(
                    getattr(self, key)._modules[m], device_ids=devices
                )
            opt = model["optimizer"]
            opt_name = "optimizer_{}".format(key)
            if "var" in opt:
                opt_name = opt["var"]
            self.optimizer_names.append(opt_name)
            model_params = getattr(self, key).parameters()
            if "args" in opt:
                setattr(
                    self, opt_name, (opt["name"](model_params, **opt["args"]))
                )
            else:
                setattr(self, opt_name, (opt["name"](model_params)))
            if "scheduler" in opt:
                sched = opt["scheduler"]
                if "args" in sched:
                    self.schedulers.append(
                        sched["name"](getattr(self, opt_name), **sched["args"])
                    )
                else:
                    self.schedulers.append(
                        sched["name"](getattr(self, opt_name))
                    )

        self.logger = Logger(
            self,
            losses_list,
            metrics_list,
            log_dir=log_dir,
            nrow=nrow,
            test_noise=test_noise,
        )

        self._store_loss_maps()
        self._store_metric_maps()