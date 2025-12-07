import torch
from torch import Tensor
from torchmetrics import Metric


# def compute_nmse_evm(y_true, y_pred):
#     nmse = np.sum(np.abs(y_true - y_pred) ** 2) / np.sum(np.abs(y_true) ** 2)
#     nmse_db = 10 * np.log10(nmse)
#     evm_rms = 100 * np.sqrt(nmse)
#     return nmse_db, evm_rms

class IQTensor:
    def __init__(self, i: Tensor, q: Tensor):
        self.i, self.q = i, q

    def __sub__(self, other):
        return IQTensor(self.i - other.i, self.q - other.q)

    def abs(self):
        return torch.sqrt(self.i * self.i + self.q * self.q)


class NMSE_dB(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("num", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("den", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = preds.squeeze(), target.squeeze()
        preds = IQTensor(preds[0], preds[1])
        target = IQTensor(target[0], target[1])

        error = target - preds
        self.num += error.abs() ** 2
        self.den += target.abs() ** 2

    def compute(self) -> Tensor:
        nmse = self.num / self.den
        nmse_db = 10 * torch.log10(nmse)
        return nmse_db


class EVM_rms(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("num", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("den", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = preds.squeeze(), target.squeeze()
        preds = IQTensor(preds[0], preds[1])
        target = IQTensor(target[0], target[1])

        error = target - preds
        self.num += error.abs() ** 2
        self.den += target.abs() ** 2

    def compute(self) -> Tensor:
        nmse = self.num / self.den
        evm_rms = 100 * torch.sqrt(nmse)
        return evm_rms
