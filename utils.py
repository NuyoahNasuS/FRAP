from copy import deepcopy
import math
import os
import pickle
import torch
from tqdm import tqdm
import json
import torch.nn as nn
from collections import Counter
import random
import torch.nn.functional as F
import time
import ot
import torch.optim as optim
import numpy as np

def get_n_classes(dataset):
    n_class = {
        'CIFAR10': 10,
        'CIFAR100': 100,
        'ImageNet': 1000,
        'TinyImageNet': 200,
        'living17': 17,
        'nonliving26': 26,
        'entity13': 13,
        'entity30': 30,
        'Camelyon17': 2,
        'Fmow': 62,
        'Rxrx1': 1139,
        'Amazon': 5,
        'CivilComments': 2,
        'MNIST': 10,
        'DomainNet': 345,
    }
    return n_class[dataset]

def get_optimizer(dataset, net, lr, pretrained):
    if dataset in ["CIFAR10", "CIFAR100", "TinyImageNet"]:
        if pretrained:
            return optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0)
        else:
            return optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif dataset in ["MNIST", "ImageNet"]:
        return optim.Adam(net.parameters(), lr=lr)
    elif dataset == "Camelyon17":
        return optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    elif dataset in ["living17", "nonliving26", "entity13", "entity30"]:
        return optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif dataset == "Fmow":
        return optim.Adam(net.parameters(), lr=lr)
    elif dataset == "Rxrx1":
        return optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    elif dataset in ['Amazon', 'CivilComments']:
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return optim.AdamW(params, lr=lr)
    elif dataset == 'DomainNet':
        return optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
def get_lr_scheduler(dataset, opt, pretrained, T_max=-1):
    if dataset in ["CIFAR10", "CIFAR100", "TinyImageNet"]:
        if pretrained:
            return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max)
        else:
            return optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 200], gamma=0.1)
    elif dataset == "MNIST":
        return optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 200], gamma=1)
    elif dataset == "ImageNet":
        return optim.lr_scheduler.MultiStepLR(opt, milestones=[100], gamma=1)
    elif dataset in ["living17", "nonliving26"]:
        return optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 50], gamma=0.1)
    elif dataset in ["entity13", "entity30"]:
        return optim.lr_scheduler.MultiStepLR(opt, milestones=[20, 40], gamma=0.1)
    elif dataset == "Camelyon17":
        return optim.lr_scheduler.MultiStepLR(opt, milestones=[100], gamma=1)
    elif dataset == "Fmow":
        return optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.96)
    elif dataset == "Rxrx1":
        return optim.lr_scheduler.OneCycleLR(
            opt, max_lr=1e-4, div_factor=1e12, pct_start=0.11, final_div_factor=1e12,
            cycle_momentum=False, base_momentum=0, max_momentum=0, total_steps=T_max
        )
    elif dataset == "Amazon":
        return optim.lr_scheduler.PolynomialLR(opt, total_iters=3, power=1)
    elif dataset == "CivilComments":
        return optim.lr_scheduler.PolynomialLR(opt, total_iters=5, power=1)
    elif dataset == "DomainNet":
        return optim.lr_scheduler.MultiStepLR(opt, milestones=[20, 30], gamma=0.1)

# ---------------------------------- Help estimation ---------------------------------- #  
def get_Results(model, dataloader, device, save_dir):
    if os.path.exists(save_dir):
        print("Load saved results from ", save_dir)
        data = pickle.load(open(save_dir, "rb"))
        preds, acts, targs = data["preds"], data["acts"], data["targs"]
        return preds.to(device), acts.to(device), targs.to(device)
    else:
        preds, acts, targs = [], [], []
        print("Compute result for ", save_dir)
        with torch.no_grad():
            for items in tqdm(dataloader):
                inputs, labels = items[0], items[1]
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                _, predicted = outputs.max(1)
                acts.append(outputs)
                preds.append(predicted)
                targs.append(labels)

        preds, acts, targs = torch.cat(preds), torch.cat(acts), torch.cat(targs)

        data = {'preds': preds.cpu(), 'acts': acts.cpu(), 'targs': targs.cpu()}
        pickle.dump(data, open(save_dir, "wb"))

    return preds, acts, targs


def get_threshold(net, iid_loader, n_class, save_result_path, metric, device):
    if metric in ["ATCMC", "ATCNE"]:
        if os.path.exists(save_result_path):
            with open(save_result_path, "r") as f:
                data = json.load(f)
                threshold = data["threshold"]
        else:
            os.makedirs(os.path.dirname(save_result_path), exist_ok=True)
            threshold = compute_t(net, iid_loader, metric, device).item()
            with open(save_result_path, "w") as f:
                json.dump({"threshold": threshold}, f)
        return threshold
    elif metric == "COTT":
        if os.path.exists(save_result_path):
            with open(save_result_path, 'r') as f:
                data = json.load(f)
                threshold = data['threshold']
        else:
            threshold = compute_cott(net, iid_loader, device)
            with open(save_result_path, "w") as f:
                json.dump({"threshold": threshold}, f)
        return threshold

def compute_t(model, iid_loader, metric, device):
    model.eval()
    misclassified = 0
    mc = []
    ne = []
    with torch.no_grad():
        for items in tqdm(iid_loader):
            inputs, labels = items[0], items[1]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(1)
            misclassified += preds.ne(labels).sum().item()
            
            ne.append(torch.sum(probs * torch.log2(probs), dim=1))
            mc.append(confs)

    ne, mc = torch.cat(ne), torch.cat(mc)
    if metric == "ATCNE":
        t = torch.sort(ne)[0][misclassified-1]
    elif metric == "ATCMC":
        t = torch.sort(mc)[0][misclassified-1] 
    return t

def compute_cott(model, iid_loader, device):
    model.eval()
    softmax_vecs = []
    preds, tars = [], []
    with torch.no_grad():
        for items in tqdm(iid_loader):
            inputs, labels = items[0], items[1]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            prob = torch.softmax(outputs, dim=1)
            prediciton = torch.argmax(outputs, dim=1)

            preds.append(prediciton.cpu())
            tars.append(labels.cpu())
            softmax_vecs.append(prob.cpu())
    preds, tars = torch.cat(preds, dim=0), torch.cat(tars, dim=0)
    softmax_vecs = torch.cat(softmax_vecs, dim=0)
    target_vecs = F.one_hot(tars)

    max_n = 10000
    if len(target_vecs) > max_n:
        print(f'sampling {max_n} out of {len(target_vecs)} validation samples...')
        torch.manual_seed(0)
        rand_inds = torch.randperm(len(target_vecs))
        tars = tars[rand_inds][:max_n]
        preds = preds[rand_inds][:max_n]
        target_vecs = target_vecs[rand_inds][:max_n]
        softmax_vecs = softmax_vecs[rand_inds][:max_n]

    print('computing assignment...')
    M = torch.cdist(target_vecs.float(), softmax_vecs, p=1)
    start = time.time()
    weights = torch.as_tensor([])
    Pi = ot.emd(weights, weights, M, numItermax=1e8)
    print(f'done. {time.time() - start}s passed')

    costs = (Pi * M.shape[0] * M).sum(1) * -1
    n_incorrect = preds.ne(tars).sum()
    t = torch.sort(costs)[0][n_incorrect - 1].item()

    return t

def get_expected_label_distribution(dataset, val_set):
    if dataset in ["Fmow", "Camelyon17", "Rxrx1", "Amazon", "CivilComments"]:
        label_counts = Counter(val_set.y_array.tolist())
        total_count = len(val_set.y_array)
        label_dist = [label_counts[i] / total_count for i in range(len(label_counts))]
        return label_dist
    config = {
        'CIFAR10': [1 / 10] * 10,
        'MNIST': [1 / 10] * 10,
        'CIFAR100': [1 / 100] * 100,
        'ImageNet': [1 / 1000] * 1000,
        'TinyImageNet': [1 / 200] * 200,
        'living17': [1 / 17] * 17,
        'nonliving26': [1 / 26] * 26,
        'entity13': [1 / 13] * 13,
        'entity30': [1 / 30] * 30,
        'DomainNet': [1 / 345] * 345
    }
    return config[dataset]


def sample_label_dist(dataset, val_set, n_class, sample_size):
    dist = get_expected_label_distribution(dataset, val_set)
    labels = []
    for i in range(n_class):
        count = int(dist[i] * sample_size)
        labels.extend([i] * count)

    remainder = sample_size - len(labels)
    r_labels = random.choices(
        list(range(n_class)),
        weights=get_expected_label_distribution(dataset, val_set),
        k=remainder
    )

    labels = labels + r_labels
    return torch.as_tensor(labels)


# -------------------------------- WF healper ----------------------------------------#
class RDumb(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.num_samples_update_1 = 0
        self.num_samples_update_2 = 0
        self.e_margin = math.log(1000) * 0.4    # hyper-parameter
        self.d_margin = 0.05
        self.current_model_probs = None

        params, param_names = collect_params(model)
        model = configure_model(model)

        self.model = model
        self.optimizer = torch.optim.SGD(params, 0.00025, momentum=0.9)

        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)
        self.total_steps = 0

    def forward(self, x):
        if self.total_steps % 1000 == 0:
            load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
            self.current_model_probs = None
        outputs = self.model(x)
        entropys = softmax_entropy(outputs)
        filter_ids_1 = torch.where(entropys < self.e_margin)[0]

        # No data satisfied
        if filter_ids_1.numel() == 0:
            self.total_steps += 1
            return outputs
        
        filtered_outputs = outputs[filter_ids_1]
        filtered_entropys = entropys[filter_ids_1]

        if self.current_model_probs is not None:
            filtered_probs = filtered_outputs.softmax(1).detach()
            cosine_similarities = F.cosine_similarity(
                self.current_model_probs.unsqueeze(dim=0).expand(filtered_probs.size(0), -1),
                filtered_probs,
                dim=1
            )

            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)[0]
            if filter_ids_2.numel() == 0:
                updated_probs = self.update_model_probs(self.current_model_probs, filtered_probs)
                self.current_model_probs = updated_probs
                self.num_samples_update_1 += filter_ids_1.size(0)
                self.total_steps += 1
                return outputs

            final_outputs = filtered_outputs[filter_ids_2]
            final_entropys = filtered_entropys[filter_ids_2]
            final_probs = filtered_probs[filter_ids_2]

            updated_probs = self.update_model_probs(self.current_model_probs, final_probs)
        else:
            filter_ids_2 = torch.arange(filtered_outputs.size(0), device=filtered_outputs.device)
            final_outputs = filtered_outputs
            final_entropys = filtered_entropys
            final_probs = filtered_outputs.softmax(1)

            updated_probs = self.update_model_probs(self.current_model_probs, final_probs)

        if final_entropys.numel() > 0:
            coeff = 1 / (torch.exp(final_entropys.clone().detach() - self.e_margin))
            weighted_entropys = final_entropys * coeff
            loss = weighted_entropys.mean()

            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

        self.num_samples_update_2 += final_entropys.size(0) if final_entropys.numel() > 0 else 0
        self.num_samples_update_1 += filter_ids_1.size(0)
        self.current_model_probs = updated_probs
        self.total_steps += 1

        return outputs

    def update_model_probs(self, current_model_probs, new_probs):
        with torch.no_grad():
            if current_model_probs is None:
                if new_probs.size(0) == 0:
                    return None
                return new_probs.mean(0)
                
            if new_probs.size(0) == 0:
                return current_model_probs
            
            alpha = 0.9
            return alpha * current_model_probs + (1 - alpha) * new_probs.mean(0)

    
def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ["weight", "bias"]:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model
    
def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


# ----------------------- GdScore ----------------------- #
def compute_GdScore(model_name, model, threshold, lr, val_loader, num_classes, norm_type, device):
    model.train()
    score_list = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    for batch_data in val_loader:
        imgs, labels = batch_data[0], batch_data[1]
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        with torch.no_grad():
            probs = F.softmax(outputs, dim=1)
            confs, preds = probs.max(1)
            preds = preds.detach()
            preds[confs<threshold] = torch.LongTensor(preds[confs<threshold].shape[0]).random_(0, num_classes).to(device)
        
        optimizer.zero_grad()
        loss = criterion(outputs, preds)
        loss.backward()
        if model_name.startswith('DenseNet'):
            weight = model.model.classifier.weight.grad
        elif model_name.startswith('ResNet'):
            weight = model.model.fc.weight.grad
        else:
            raise ValueError(f"Error compute GdScore: {model_name}'s grad haven't supported!")
        score = torch.norm(weight, p=norm_type)
        score_list.append(score)
    scores = torch.Tensor(score_list).numpy()
    return scores.mean()











