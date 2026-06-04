import time
import numpy as np
import os
import torch
from Calibrate import calibrate
import math
import random
import torch.nn.functional as F
from utils import sample_label_dist, get_threshold, get_Results, compute_GdScore, RDumb
import ot
import pandas as pd
from model_load import get_cv_model, initializa_bert_based_model

class HistogramDensity:
    def _histedges_equalN(self, x, nbin):
        npt = len(x)
        return np.interp(
            np.linspace(0, npt, nbin+1),
            np.arange(npt),
            np.sort(x)
        )
    
    def __init__(self, num_bins=10, equal_mass=False):
        self.num_bins = num_bins
        self.equal_mass = equal_mass

    def fit(self, vals):
        if self.equal_mass:
            self.bins = self._histedges_equalN(vals, self.num_bins)
        else:
            self.bins = np.linspace(0, 1.0, self.num_bins+1)
        self.bins[0] = 0.0
        self.bins[self.num_bins] = 1.0

        self.hist, bin_edges = np.histogram(vals, bins=self.bins, density=True)

    def density(self, x):
        cur_bins = np.digitize(x, self.bins, right=True)

        cur_bins -= 1
        return self.hist[cur_bins]
    
def get_im_estimate(iid_acts, ood_acts, corr_source):
    probs_source, probs_target = iid_acts.max(1)[0], ood_acts.max(1)[0]
    probs_source = probs_source.numpy()
    probs_target = probs_target.numpy()
    corr_source = corr_source.numpy()

    source_binning = HistogramDensity()
    source_binning.fit(probs_source)
    target_binning = HistogramDensity()
    target_binning.fit(probs_target)

    weights = target_binning.density(probs_source) / source_binning.density(probs_source)
    weights = weights / np.mean(weights)

    return 1 - np.mean(weights * corr_source)

def get_AC_estimate(ood_acts):
    max_conf = torch.max(ood_acts, dim=-1)[0]
    est = 1 - torch.mean(max_conf).item()
    return est

def get_DoC_estimate(iid_acts, ood_acts, iid_preds, iid_targs):
    source_conf = iid_acts.max(1)[0]
    target_conf = ood_acts.max(1)[0]
    source_err = (iid_preds!=iid_targs).sum().item() / len(iid_targs)
    est = source_err + torch.mean(source_conf).item() - torch.mean(target_conf).item()
    return est

def get_GDE_estimate(model_name, model_seed, pretrained, ood_preds, alt_ckpt_path, save_altCalibrate_dir, save_altCalibrate_name, save_altClibrate_Result_path, n_class, id_val_loaders, ood_loaders, device, opt_bias=False):
    if not os.path.exists(alt_ckpt_path):
        print(f"GDE_estimation Error: No such alter Model Path :{alt_ckpt}")
        exit()
    
    if model_name == 'distilbert-base-uncased':
        alt_model = initializa_bert_based_model(n_class)
    else:
        alt_model = get_cv_model(model_name, n_class, model_seed, pretrained)
    
    alt_ckpt = torch.load(alt_ckpt_path, map_location=device)
    alt_model.load_state_dict(alt_ckpt['model_state_dict'])
    alt_model.to(device)
    alt_model.eval()

    os.makedirs(save_altCalibrate_dir, exist_ok=True)
    temp_file = os.path.join(save_altCalibrate_dir, save_altCalibrate_name)
    alt_model = calibrate(alt_model, n_class, id_val_loaders, temp_file, opt_bias)
    alt_preds, _, _ = get_Results(alt_model, ood_loaders, device, save_altClibrate_Result_path)
    est = alt_preds.ne(ood_preds).sum().item() / len(alt_preds)
    return est

def get_ATCNE_estimate(ood_acts, model, iid_loader, n_class, save_result_dir, device):
    threshold = get_threshold(model, iid_loader, n_class, save_result_dir, "ATCNE", device)
    ne = torch.sum(ood_acts * torch.log2(ood_acts), dim=1)
    est = (ne < threshold).sum().item() / len(ood_acts)
    cost_dist = torch.sort(ne)[0].tolist()
    return est

def get_ATCMC_estimate(ood_acts, model, iid_loader, n_class, save_result_path, device):
    threshold = get_threshold(model, iid_loader, n_class, save_result_path, "ATCMC", device)
    mc = ood_acts.max(1)[0]
    est = (mc < threshold).sum().item() / len(ood_acts)
    cost_dist = torch.sort(mc)[0].tolist()
    return est

def get_COT_estimate(ood_acts, dataset, val_set, n_test_sample, n_class):
    batch_size = min(10000, n_test_sample)
    n_batch = math.ceil(n_test_sample / batch_size)
    print(f'total of {n_test_sample} test samples, running {n_batch} batches.')
    random.seed(10)
    if n_batch > 1:
        est = 0
        for _ in range(n_batch):
            rand_inds = torch.as_tensor(random.choices(list(range(n_test_sample)), k=batch_size))
            iid_acts_batch = F.one_hot(sample_label_dist(dataset, val_set, n_class, batch_size))
            ood_acts_batch = ood_acts[rand_inds]

            # distance map or cost matrix
            M = torch.cdist(iid_acts_batch.float(), ood_acts_batch, p=1)
            weights = torch.as_tensor([])
            # emd2 ：return the total cost
            est += (ot.emd2(weights, weights, M, numItermax=1e8, numThreads=8) / 2).item()
        est = est / n_batch
    else:
        iid_acts = F.one_hot(sample_label_dist(dataset, val_set, n_class, len(ood_acts)))
        M = torch.cdist(iid_acts.float(), ood_acts, p=1) / 2
        weights = torch.as_tensor([])
        # emd : return the tansport matrix
        Pi = ot.emd(weights, weights, M, numItermax=1e8)

        costs = (Pi * M.shape[0] * M).sum(1)
        est = costs.mean().item()
    return est

def get_COTT_estimate(model, ood_acts, dataset, val_set, iid_loader, n_class, save_result_dir, n_test_sample, device):
    threshold = get_threshold(model, iid_loader, n_class, save_result_dir, "COTT", device)
    batch_size = min(10000, n_test_sample)
    n_batch = math.ceil(n_test_sample / batch_size)
    print(f'total of {n_test_sample} test samples, running {n_batch} batches.')
    
    random.seed(10)
    if n_batch > 1:
        est = 0
        cost_dist = []
        for _ in range(n_batch):
            rand_inds = torch.as_tensor(random.choices(list(range(n_test_sample)), k=batch_size))
            ood_acts_batch = ood_acts[rand_inds]

            iid_acts_batch = F.one_hot(sample_label_dist(dataset, val_set, n_class, batch_size))

            M = torch.cdist(iid_acts_batch.float(), ood_acts_batch, p=1)

            weights = torch.as_tensor([])
            Pi = ot.emd(weights, weights, M, numItermax=1e8)

            costs = (Pi * M.shape[0] * M).sum(1) * -1

            est += (costs < threshold).sum().item() / batch_size
            cost_dist.append(costs)
        
        est = est / n_batch
        cost_dist = torch.sort(torch.cat(cost_dist, dim=0))[0].tolist()
    else:
        iid_acts = F.one_hot(sample_label_dist(dataset, val_set, n_class, n_test_sample))

        M = torch.cdist(iid_acts.float(), ood_acts, p=1)
        weights = torch.as_tensor([])
        Pi = ot.emd(weights, weights, M, numItermax=1e8)

        costs = (Pi * M.shape[0] * M).sum(1) * -1
        est = (costs < threshold).sum().item() / batch_size
        cost_dist = torch.sort(costs)[0].tolist()

    return est

def get_ProjNorm_estimate(ood_preds_dist, iid_preds_dist, iid_acc):
    est = min(
        sum(abs(np.array(ood_preds_dist) - np.array(iid_preds_dist))) / 2 + (1 - iid_acc), 1
    )
    return est

def get_GdScore_estimate(model_name, model, val_loader, num_classes, device, lr=0.001, threshold=0.5, norm_type=0.3):
    score = compute_GdScore(model_name, model, threshold, lr, val_loader, num_classes, norm_type, device)
    score = float(score)

    return score

# ----------------------------------------------------- WF -------------------------------------------------------------------#
def get_WF(model, test_loader, test_size, test_acc, device):
    model = RDumb(model)
    stored_images, stored_labels = get_unlabeled_subset(model.model, test_loader, test_size, device)

    iteration = 0
    df = None

    while iteration <= 990:
        for i, (batch) in enumerate(test_loader):
            images, labels = batch[0], batch[1]

            if iteration == 0 or iteration == 990:
                df = check_flips(model.model, stored_images, stored_labels, iteration, device, df)
            
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            _ = model(images)
            iteration += 1

    coefficients = [3.55553868e-04, -3.21445023e-01, 7.56618580e+01]
    coefficients_2 = [-0.10318227, 83.55158891]
    coefficients_3 = [-1.01165493e-05, -9.59406999e-02, 8.25756446e+01]
    f = np.poly1d(coefficients)
    f2 = np.poly1d(coefficients_2)
    f3 = np.poly1d(coefficients_3)

    weighted_flips = count_flips(df)
    unweighted_flips = count_flips(df, weighted=False)

    estimated_accuracy = f(weighted_flips)
    estimated_accuracy_2 = f2(unweighted_flips)
    estimated_accuracy_3 = f3(unweighted_flips)

    mae1 = abs(float(estimated_accuracy / 100) - test_acc)
    mae2 = abs(float(estimated_accuracy_2 / 100) - test_acc)
    mae3 = abs(float(estimated_accuracy_3 / 100) - test_acc)
    mae_values = [mae1, mae2, mae3]

    est = min(mae_values)
    return est

def get_unlabeled_subset(model, loader, test_size, device):
    total_images = min(test_size, 1000)

    model.eval()

    stored_images = torch.empty((total_images, *next(iter(loader))[0].shape[1:]))
    stored_labels = torch.empty(total_images, dtype=torch.long)

    current_idx = 0

    with torch.no_grad():
        for batch_data in loader:
            images, _ = batch_data[0], batch_data[1]

            batch_size_actual = min(images.size(0), total_images - current_idx)
            if batch_size_actual < 0:
                break

            images_gpu = images[:batch_size_actual].to(device, non_blocking=True)
            output = model(images_gpu)

            _, pred = output.max(dim=1)

            stored_images[current_idx:current_idx + batch_size_actual] = images[:batch_size_actual]
            stored_labels[current_idx:current_idx + batch_size_actual] = pred.cpu()

            current_idx += batch_size_actual

            if current_idx >= total_images:
                break

    stored_images = stored_images[:current_idx]
    stored_labels = stored_labels[:current_idx]

    model.train()
    return stored_images, stored_labels

def check_flips(model, stored_images, stored_labels, iteration, device, df=None):
    batch_size = 256
    model.eval()

    iteration_list = []
    image_index_list = []
    top_class_list = []
    confidence_list = []

    total_seen_so_far = 0

    with torch.no_grad():
        for i in range(0, stored_images.size(0), batch_size):
            images = stored_images[i: i + batch_size].to(device, non_blocking=True)

            output = model(images)
            probs = F.softmax(output, dim=1)

            top_vals, top_indices = torch.max(probs, dim=1)

            cur_batch_num = images.size(0)
            iteration_list.extend([iteration] * cur_batch_num)
            image_index_list.extend(range(total_seen_so_far, total_seen_so_far + cur_batch_num))

            top_class_list.extend(top_indices.cpu().numpy().tolist())
            confidence_list.extend(top_vals.cpu().numpy().tolist())

            total_seen_so_far += cur_batch_num

    new_predictions_df = pd.DataFrame({
        'Iteration': iteration_list,
        'Image_Index': image_index_list,
        'Top_Class': top_class_list,
        'Confidence': confidence_list,
    })

    model.train()

    if df is not None:
        return pd.concat([df, new_predictions_df], ignore_index=True)
    else:
        return new_predictions_df

def count_flips(df, end=990, weighted=True):
    if df.empty:
        return None
    
    start = 0
    df_filtered = df[df['Iteration'].isin([start, end])].copy()

    if df_filtered.empty:
        return None
    
    if end not in df_filtered['Iteration'].values:
        return None
    
    idx = df_filtered.groupby(['Image_Index', 'Iteration'])['Confidence'].idxmax()
    df_selected = df_filtered.loc[idx].copy()

    df_pivot = df_selected.pivot(index='Image_Index', columns='Iteration', values=['Top_Class', 'Confidence'])

    if isinstance(df_pivot.columns, pd.MultiIndex):
        top_class_cols = df_pivot['Top_Class']
        confidence_cols = df_pivot['Confidence'] if 'Confidence' in df_pivot.columns.get_level_values(0) else None
    else:
        df_pivot = df_selected.pivot(index='Image_Index', columns='Iteration', values='Top_Class')
        if weighted:
            confidence_pivot = df_selected.pivot(index='Image_Index', columns='Iteration', values='Confidence')
    
    if isinstance(df_pivot.columns, pd.MultiIndex):
        label_flipped = (top_class_cols[start] != top_class_cols[end]).astype(int)
    else:
        label_flipped = (df_pivot[start] != df_pivot[end]).astype(int)

    if weighted:
        if isinstance(df_pivot.columns, pd.MultiIndex):
            initial_confidence = confidence_cols[start]
        else:
            initial_confidence = confidence_pivot[start]
        
        weight = initial_confidence.rank() / len(initial_confidence)
    else:
        weight = 1
    
    weighted_sum = (label_flipped * weight).sum()
    return weighted_sum




