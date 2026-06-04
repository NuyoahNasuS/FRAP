import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import torch
from utils import get_n_classes, get_Results, get_expected_label_distribution
from data_loader import *
from Calibrate import calibrate
import torch.nn.functional as F
from collections import Counter
import time
from baseline import *
import json
from model_load import get_cv_model, initializa_bert_based_model
# RAP
# from reference_aligned_pred_V3 import *
# FRAP
from reference_aligned_pred import *


def main():
    parser = argparse.ArgumentParser(description='Estimate target domain performance.')
    parser.add_argument('--model', default='ResNet18', type=str)
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--metric', default='IM', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--model_seed', default=1, type=int)
    parser.add_argument('--ckpt_epoch', default=20, type=int)

    # synthetic shifts configs
    parser.add_argument('--data_path', default='/home/shuxuan/public_data/Estimate_Performance', type=str)
    
    args = parser.parse_args()
    print(vars(args))

    dataset_name = args.dataset
    batch_size = args.batch_size
    data_path = args.data_path

    label2class = None

    if dataset_name == "CIFAR10":
        _, _, val_set, val_loader, test_sets, test_loaders, test_types = get_cifar10(batch_size, data_path, False, True, True)
    elif dataset_name == "CIFAR100":
        _, _, val_set, val_loader, test_sets, test_loaders, test_types = get_cifar100(batch_size, data_path, False, True, True)
    elif dataset_name == "MNIST":
        _, _, val_set, val_loader, test_sets, test_loaders, test_types = get_mnist(batch_size, data_path, False, True, True)
    elif dataset_name == "ImageNet":
        _, _, val_set, val_loader, test_sets, test_loaders, test_types = get_ImageNet(batch_size, data_path, False, True, True)
    elif dataset_name == "TinyImageNet":
        _, _, val_set, val_loader, test_sets, test_loaders, test_types = get_TinyImageNet(batch_size, data_path, False, True, True)
    elif dataset_name in ["living17", "entity13", "entity30", "nonliving26"]:
        _, _, val_set, val_loader, test_sets, test_loaders, test_types, label2class = get_imagenet_breeds(batch_size, data_path, dataset_name, False, True, True)
    elif dataset_name == "Camelyon17":
        _, _, val_set, val_loader, test_sets, test_loaders, test_types = get_camelyon17(batch_size, data_path, False, True, True)
    elif dataset_name == "Fmow":
        _, _, val_set, val_loader, test_sets, test_loaders, test_types = get_fmow(batch_size, data_path, False, True, True)
    elif dataset_name == "Rxrx1":
        _, _, val_set, val_loader, test_sets, test_loaders, test_types = get_rxrx1(batch_size, data_path, False, True, True)
    elif dataset_name == "Amazon":
        _, _, val_set, val_loader, test_sets, test_loaders, test_types = get_amazon(batch_size, data_path, 'distilbert-base-uncased', 512, False, True, True)
    elif dataset_name == "CivilComments":
        _, _, val_set, val_loader, test_sets, test_loaders, test_types = get_civilcomments(batch_size, data_path, 'distilbert-base-uncased', 300, False, True, True)
    elif dataset_name == "DomainNet":
        _, _, val_set, val_loader, test_sets, test_loaders, test_types = get_domainnet(batch_size, data_path, False, True, True)

    size = len(test_types)
    for i in range(size):
        test_set, test_loader = test_sets[i], test_loaders[i]
        test_type = test_types[i]
        estimate(args, val_set, val_loader, test_set, test_loader, test_type, label2class)


def estimate(args, val_set, val_loader, test_set, test_loader, test_type, label2class=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = args.metric
    model_name = args.model
    pretrained = args.pretrained
    model_seed = args.model_seed
    model_epoch = args.ckpt_epoch
    dataset_name = args.dataset

    n_class = get_n_classes(dataset_name)

    if pretrained:
        save_gather_dir = f"/home/shuxuan/public_data/Estimate_Performance/checkpoints/{dataset_name}/{model_name}/pretrained_ts"
    else:
        save_gather_dir = f"/home/shuxuan/public_data/Estimate_Performance/checkpoints/{dataset_name}/{model_name}/scratch_ts"
    os.makedirs(save_gather_dir, exist_ok=True)
    save_id_gather_path = f"{save_gather_dir}/id_Seed{model_seed}_epoch{model_epoch}.pkl"
    save_ood_gather_path = f"{save_gather_dir}/ood_Seed{model_seed}_epoch{model_epoch}_Data_{test_type}.pkl"

    if pretrained:
        model_dir = f"/home/shuxuan/public_data/Estimate_Performance/checkpoints/{dataset_name}/{model_name}/pretrained"
    else:
        model_dir = f"/home/shuxuan/public_data/Estimate_Performance/checkpoints/{dataset_name}/{model_name}/scratch"

    if model_name == 'distilbert-base-uncased':
        model = initializa_bert_based_model(n_class)
    else:
        model = get_cv_model(model_name, n_class, model_seed, pretrained)

    ckpt_path = f"{model_dir}/seed{model_seed}_epoch{model_epoch}.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    print("Calibrating models...")
 
    temperature_path = f"{save_gather_dir}/seed{model_seed}_epoch{model_epoch}_temperature.json"

    model = calibrate(model, n_class, val_loader, temperature_path, device)
    print("Calibrating done!")

    iid_preds, iid_acts, iid_targs = get_Results(model, val_loader, device, save_id_gather_path)
    ood_preds, ood_acts, ood_targs = get_Results(model, test_loader, device, save_ood_gather_path)

    iid_acts = F.softmax(iid_acts, dim=1).cpu()
    ood_acts = F.softmax(ood_acts, dim=1).cpu()

    iid_acc = ((iid_preds == iid_targs).sum() / len(iid_targs)).item()
    ood_acc = ((ood_preds == ood_targs).sum() / len(ood_targs)).item()

    iid_confs = iid_acts.amax(dim=1).mean().item()

    print('------------------')
    print('validation acc:', iid_acc)
    print('validation confidence:', iid_confs)
    print('confidence gap:', iid_confs - iid_acc)
    print('------------------')
    print()

    ood_preds_count = Counter(ood_preds.tolist())
    ood_targs_count = Counter(ood_targs)

    iid_preds_count = Counter(iid_targs.tolist())

    iid_targs_dist = get_expected_label_distribution(dataset_name, val_set)
    ood_targs_dist = [ood_targs_count[i] / len(ood_acts) for i in range(n_class)]
    ood_preds_dist = [ood_preds_count[i] / len(ood_acts) for i in range(n_class)]
    iid_preds_dist = [iid_preds_count[i] / len(iid_acts) for i in range(n_class)]

    print('------------------')
    print("ood real label tv:", sum(abs(np.array(ood_targs_dist) - np.array(iid_targs_dist))) / 2 )
    print("ood pseudo label tv:", sum(abs(np.array(ood_preds_dist) - np.array(iid_preds_dist))) / 2 )
    print("ood pseudo-real label tv:", sum(abs(np.array(ood_preds_dist) - np.array(ood_targs_dist))) / 2 )
    print('------------------')
    print()

    start = time.time()

    if metric == "AC":
        est = get_AC_estimate(ood_acts)
    elif metric == "DoC":
        est = get_DoC_estimate(iid_acts, ood_acts, iid_preds, iid_targs)
    elif metric == "IM":
        est = get_im_estimate(iid_acts, ood_acts, (iid_preds == iid_targs).cpu())
    elif metric == "GDE":
        seeds = [0, 1, 10]
        remaining_seeds = [s for s in seeds if s != model_seed]
        alt_model_seed = random.choice(remaining_seeds)
        alt_ckpt_path = f"{model_dir}/seed{alt_model_seed}_epoch{model_epoch}.pt"
        
        if pretrained:
            alt_save_gather_dir = f"/home/shuxuan/public_data/Estimate_Performance/checkpoints/{dataset_name}/{model_name}/pretrained_ts"
        else:
            alt_save_gather_dir = f"/home/shuxuan/public_data/Estimate_Performance/checkpoints/{dataset_name}/{model_name}/scratch_ts"
        alt_save_calibrate_name = f"seed{alt_model_seed}_epoch{model_epoch}_temperature.json"
        alt_save_ood_gather_path = f"{alt_save_gather_dir}/ood_Seed{alt_model_seed}_epoch{model_epoch}_Data_{test_type}.pkl"
        est = get_GDE_estimate(model_name, model_seed, pretrained, ood_preds, alt_ckpt_path, alt_save_gather_dir, alt_save_calibrate_name, alt_save_ood_gather_path, n_class, val_loader, test_loader, device)
    elif metric == "ATC_MC":
        if pretrained:
            save_threshold_path = f"/home/shuxuan/public_data/Estimate_Performance/checkpoints/{dataset_name}/{model_name}/pretrained_ts_{metric}_Seed{model_seed}_epoch{model_epoch}_threshold.json"
        else:
            save_threshold_path = f"/home/shuxuan/public_data/Estimate_Performance/checkpoints/{dataset_name}/{model_name}/scratch_ts_{metric}_Seed{model_seed}_epoch{model_epoch}_threshold.json"
        est = get_ATCMC_estimate(ood_acts, model, val_loader, n_class, save_threshold_path, device)
    elif metric == "ATC_NE":
        if pretrained:
            save_threshold_path = f"/home/shuxuan/public_data/Estimate_Performance/checkpoints/{dataset_name}/{model_name}/pretrained_ts_{metric}_Seed{model_seed}_epoch{model_epoch}_threshold.json"
        else:
            save_threshold_path = f"/home/shuxuan/public_data/Estimate_Performance/checkpoints/{dataset_name}/{model_name}/scratch_ts_{metric}_Seed{model_seed}_epoch{model_epoch}_threshold.json"
        est = get_ATCNE_estimate(ood_acts, model, val_loader, n_class, save_threshold_path, device)
    elif metric == "ProjNorm":
        est = get_ProjNorm_estimate(ood_preds_dist, iid_preds_dist, iid_acc)
    elif metric == "COT":
        est = get_COT_estimate(ood_acts, dataset_name, val_set, len(test_set), n_class)
    elif metric == "COTT":
        if pretrained:
            save_threshold_path = f"/home/shuxuan/public_data/Estimate_Performance/checkpoints/{dataset_name}/{model_name}/pretrained_ts_{metric}_Seed{model_seed}_epoch{model_epoch}_threshold.json"
        else:
            save_threshold_path = f"/home/shuxuan/public_data/Estimate_Performance/checkpoints/{dataset_name}/{model_name}/scratch_ts_{metric}_Seed{model_seed}_epoch{model_epoch}_threshold.json"
        est = get_COTT_estimate(model, ood_acts, dataset_name, val_set, val_loader, n_class, save_threshold_path, len(test_set), device)
    elif metric == "GdScore":
        val_GdScore = get_GdScore_estimate(model_name, model, val_loader, n_class, device, lr=0.001, threshold=0.5, norm_type=0.3)
        test_GdScore = get_GdScore_estimate(model_name, model, test_loader, n_class, device, lr=0.001, threshold=0.5, norm_type=0.3)
    elif metric == "WF":
        est_mae = get_WF(model, test_loader, len(test_set), ood_acc, device)
    elif metric == "FRAP":
        class_list = get_label_text(dataset_name, val_set, label2class)
        if dataset_name == 'MNIST':
            clip_model, text_features = get_reference_model(device, class_list, True)
        else:
            clip_model, text_features = get_reference_model(device, class_list)
        
        val_calibrate_temperature, _, _ = calibrate_clip(dataset_name, iid_acts, clip_model, val_loader, text_features, device, epochs=50, lr=0.01)
        test_calibrate_temperature, cached_clip_logits, cached_labels = calibrate_clip(dataset_name, ood_acts, clip_model, test_loader, text_features, device, epochs=1000, lr=0.01)
        if pretrained:
            save_threshold_path = f"/home/shuxuan/public_data/Estimate_Performance7/checkpoints/{dataset_name}/{model_name}/pretrained_ts_{metric}_Seed{model_seed}_epoch{model_epoch}_threshold.json"
        else:
            save_threshold_path = f"/home/shuxuan/public_data/Estimate_Performance7/checkpoints/{dataset_name}/{model_name}/scratch_ts_{metric}_Seed{model_seed}_epoch{model_epoch}_threshold.json"
        est = get_FRAP(val_calibrate_temperature, test_calibrate_temperature, dataset_name, model, clip_model, val_loader, text_features, ood_acts, cached_clip_logits, save_threshold_path, device, cached_labels)

    if metric == "GdScore":
        print('------------------')
        print('True OOD error:', 1 - ood_acc)
        print(f'{metric} val score: {val_GdScore}, val accuracy: {iid_acc}')
        print(f'{metric} test score: {test_GdScore}, test accuracy: {ood_acc}')
        print(f'Time: {time.time() - start}')
        print('------------------')
        print()
    elif metric not in ["WF", "GdScore", "FRAP"]:
        print('------------------')
        print('True OOD error:', 1 - ood_acc)
        print(f'{metric} predicted OOD error:', est)
        print(f'MAE: {abs(1 - ood_acc - est)}')
        print(f'Time: {time.time() - start}')
        print('------------------')
        print()
    elif metric == "WF":
        print('------------------')
        print('True OOD error:', 1 - ood_acc)
        print(f'MAE: {est_mae}')
        print(f'Time: {time.time() - start}')
        print('------------------')
        print()
    elif metric == "FRAP":
        print('------------------')
        print('True OOD error:', 1 - ood_acc)
        print(f'{metric} predicted OOD error:', est)
        print(f'MAE: {abs(1 - ood_acc - est)}')
        print(f'Time: {time.time() - start}')
        print('------------------')
        print()

    test_category = ""
    if "same" in test_type:
        test_category = "Same"
    elif "novel" in test_type:
        test_category = "Novel"
    elif "corruption" in test_type:
        test_category = "Synthetic"
    elif test_type.startswith("domain"):
        test_category = test_type.split('_')[1]
    else:
        test_category = "Natural"


    if pretrained:
        result_path = f"/home/shuxuan/public_data/Estimate_Performance7/results/{dataset_name}/pretrained_ts/{metric}/{test_category}/{model_name}_seed{model_seed}_epoch{model_epoch}.json"
    else:
        result_path = f"/home/shuxuan/public_data/Estimate_Performance7/results/{dataset_name}/scratch_ts/{metric}/{test_category}/{model_name}_seed{model_seed}_epoch{model_epoch}.json"

    if metric == "GdScore":
        result_path = f"/home/shuxuan/public_data/Estimate_Performance/results/_GdScore_result/{dataset_name}_{model_name}_seed{model_seed}_epoch{model_epoch}.json"

    print(result_path)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    if not os.path.exists(result_path):
        with open(result_path, "w") as f:
            json.dump([], f)

    with open(result_path, 'r') as f:
        data = json.load(f)
    
    if metric == "FRAP":
        data.append({
            'pretrained': pretrained,
            'test_type': test_type,
            'metric type': metric,
            'metric value': float(est),
            'acc': float(ood_acc),
            'error': 1 - ood_acc,
            'MAE': abs(1 - ood_acc - est),
        })
    elif metric == "WF":
        data.append({
            'pretrained': pretrained,
            'test_type': test_type,
            'metric type': metric,
            'metric value': float(est_mae),     # current testdata's MAE
            'acc': float(ood_acc),
            'error': 1 - ood_acc,
            'MAE': est_mae
        })
    elif metric == "GdScore":
        data.append({
            'dataset': dataset_name,
            'model': f"{model_name}_seed{model_seed}_epoch{model_epoch}",
            'pretrained': pretrained,
            'test_type': test_type,
            'test_category': test_category,
            'metric type': metric,
            'val_GdScore': float(val_GdScore),
            'val_acc': float(iid_acc),
            'test_GdScore': float(test_GdScore),
            'test_acc': float(ood_acc),
            'error': 1 - ood_acc,
        })
    else:
        data.append({
            'pretrained': pretrained,
            'test_type': test_type,
            'metric type': metric,
            'metric value': float(est),
            'acc': float(ood_acc),
            'error': 1 - ood_acc,
            'MAE': abs(1 - ood_acc - est),
        })

    with open(result_path, 'w') as f:
        json.dump(data, f)
    
if __name__ == "__main__":
    main()




