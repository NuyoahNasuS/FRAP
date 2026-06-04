import torch.nn as nn
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch import optim
import os
import json

class ModelWithTemperature(nn.Module):
    def __init__(self, model, n_class, opt_bias=False, device=None):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.device = device
        self.log_temperature = nn.Parameter(torch.zeros(1))
        self.opt_bias = opt_bias
        if opt_bias:
            self.bias = nn.Parameter(torch.zeros(1, n_class))
        else:
            self.register_parameter("bias", None)
        
        self.to(device)

    @property
    def temperature(self):
        return torch.exp(self.log_temperature)
    
    def forward(self, x):
        logits = self.model(x)
        return self.temperature_scale(logits)
        
    def temperature_scale(self, logits):
        scaled_logits = logits / self.temperature
        if self.bias is not None:
            scaled_logits += self.bias
        return scaled_logits
    
    def set_temperature(self, temp, bias=None):
        self.log_temperature.data = torch.log(torch.tensor(temp, device=self.device))
        if self.bias is not None and bias is not None:
            self.bias.data = torch.as_tensor(bias, device=self.device)

    def find_temperature(self, valid_loader, max_iter=100000, lr=0.01):
        self.eval()
        logits_list, labels_list = [], []

        with torch.no_grad():
            for items in tqdm(valid_loader, desc="Collecting logits"):
                inputs, labels = items[0], items[1]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = _SoftBinned_ECELoss().to(self.device)

        before_nll = nll_criterion(logits, labels)
        before_ece = ece_criterion(logits, labels)
        print(f"[Before] NLL: {before_nll:.3f}, ECE: {before_ece:.3f}")

        if self.opt_bias:
            params = [self.log_temperature, self.bias]
        else:
            params = [self.log_temperature]

        optimizer = optim.LBFGS(params, lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            nll_loss = nll_criterion(self.temperature_scale(logits), labels)
            ece_loss = ece_criterion(self.temperature_scale(logits), labels)
            # may collapse under certain setting. Turn off the ece_loss
            loss = nll_loss + ece_loss
            loss.backward()
            return loss
        
        optimizer.step(eval_step)

        temperatured_logits = self.temperature_scale(logits)
        after_nll = nll_criterion(temperatured_logits, labels).item()
        after_ece = ece_criterion(temperatured_logits, labels).item()
        print(f"[After] Temp: {self.temperature.item():.3f}, NLL: {after_nll:.3f}, ECE: {after_ece:.3f}")

        conf = torch.softmax(temperatured_logits, dim=1).max(dim=1).values.mean().item()
        acc = (torch.argmax(temperatured_logits, dim=1) == labels).float().mean().item()
        print(f"TS Conf: {conf:.3f}, TS Acc: {acc:.3f}")

        return self


class _SoftBinned_ECELoss(nn.Module):
    def __init__(self, n_bins=15, sigma=0.05, eps=1e-8):
        super(_SoftBinned_ECELoss, self).__init__()
        self.n_bins = n_bins
        self.sigma = sigma
        self.eps = eps

        self.bin_centers = torch.linspace(1.0 / (2 * n_bins), 1.0 - 1.0 / (2 * n_bins), n_bins)
        self.register_buffer('bin_centers_tensor', self.bin_centers.unsqueeze(0))   # (1, b)

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, dim=1)      # (B,)

        accuracy = predictions.eq(labels).float()       # (B,)
        # Gaussian kernel function for soft binning
        confidences_expanded = confidences.unsqueeze(1)     # (B, 1)
        # (1/Distance between sample's confidence and the bin centers) as weight
        diffs = (confidences_expanded - self.bin_centers_tensor) / self.sigma       # (B, b)
        weights_soft = torch.exp(-0.5 * (diffs ** 2))
        weights_soft = weights_soft / (torch.sum(weights_soft, dim=1, keepdim=True) + self.eps)     # (B, b)
        bin_soft_accuracy = torch.sum(weights_soft * accuracy.unsqueeze(1), dim=0)              # (b,)
        bin_soft_confidence = torch.sum(weights_soft * confidences_expanded, dim=0)
        bin_soft_counts = torch.sum(weights_soft, dim=0)
        # Soft ECE
        bin_soft_counts_normalized = bin_soft_counts / (bin_soft_counts.sum() + self.eps)
        soft_ece = torch.sum(bin_soft_counts_normalized * torch.abs(bin_soft_accuracy - bin_soft_confidence))
        return soft_ece


def calibrate(model, n_class, val_loader, temp_file, device=None, opt_bias=False):
    device = device or torch.device('cpu')
    wrapper = ModelWithTemperature(model, n_class, opt_bias, device)

    if os.path.exists(temp_file):
        with open(temp_file) as f:
            params = json.load(f)
        wrapper.set_temperature(params["temperature"], params["bias"])
    else:
        wrapper.find_temperature(val_loader)
        params = {
            "temperature": wrapper.temperature.item(),
            "bias": (
                wrapper.bias.detach().cpu().tolist()
                if wrapper.bias is not None
                else None
            )
        }
        with open(temp_file, "w") as f:
            json.dump(params, f)

    return wrapper








