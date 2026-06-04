import math
import pickle
import clip
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import json


def get_wordnetId(data_name, dataset=None):
    if data_name in ["ImageNet", "TinyImageNet"]:
        return dataset.classes
    else:
        raise ValueError(f"{data_name} Not Support")

def get_label_text(data_name, dataset=None, label_classes=None):
    if data_name in ['CIFAR10', 'CIFAR100']:
        return dataset.classes
    elif data_name == 'ImageNet':
        ImageNet_classList_file = r'/home/shuxuan/public_data/Estimate_Performance/ImageNet_class.pkl'
        with open(ImageNet_classList_file, 'rb') as f:
            class_list = pickle.load(f)
        return class_list
    elif data_name == 'TinyImageNet':
        TinyImageNet_classList_file = r'/home/shuxuan/public_data/Estimate_Performance/TinyImageNet_class.pkl'
        with open(TinyImageNet_classList_file, 'rb') as f:
            class_list = pickle.load(f)
        return class_list
    elif data_name == 'Fmow':
        categories = ["airport", "airport hangar", "airport terminal", "amusement park", "aquaculture", "archaeological site", "barn", "border checkpoint", "burial site", "car dealership", "construction site", "crop field", "dam", "debris or rubble", "educational institution", "electric substation", "factory or powerplant", "fire station", "flooded road", "fountain", "gas station", "golf course", "ground transportation station", "helipad", "hospital", "impoverished settlement", "interchange", "lake or pond", "lighthouse", "military facility", "multi-unit residential", "nuclear powerplant", "office building", "oil or gas facility", "park", "parking lot or garage", "place of worship", "police station", "port", "prison", "race track", "railway bridge", "recreational facility", "road bridge", "runway", "shipyard", "shopping mall", "single-unit residential", "smokestack", "solar farm", "space facility", "stadium", "storage tank", "surface mine", "swimming pool", "toll booth", "tower", "tunnel opening", "waste disposal", "water treatment facility", "wind farm", "zoo"]
        return categories
    elif data_name in ["living17", "nonliving26", "entity13", "entity30"]:
        class_list = [v for k, v in label_classes.items()]
        class_list = [v.split(',')[0] for v in class_list]
        return class_list
    elif data_name == 'MNIST':
        class_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        return class_list
    elif data_name == "DomainNet":
        class_list = dataset.classes
        new_class_list = [c.replace('_', ' ') for c in class_list]
        return new_class_list
    else:
        raise ValueError(f"Get class names: No such dataset {data_name}")
    
def get_reference_model(device, classes, MNIST=False):
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    text_inputs = clip.tokenize([f"a photo of a {label}" for label in classes]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    return clip_model, text_features

def calibrate_clip(dataset, ood_probs, clip_model, test_loader, text_features, device, epochs=50, lr=0.01):
    log_temp = torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=True)
    optimizer = torch.optim.Adam([log_temp], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    cached_clip_logits, cached_labels = [], []
        
    total_samples = 0
    batch_size = test_loader.batch_size
    with torch.no_grad():
        for items in test_loader:
            images, labels = items[0], items[1]
            images, labels = images.to(device), labels.to(device)
            if dataset in ["CIFAR10", "CIFAR100", "MNIST", "TinyImageNet"]:
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            elif dataset in ["ImageNet", "Fmow", "living17", "nonliving26", "entity13", "entity30", "DomainNet"]:
                pass
            # if images.shape[1] == 1:
            #     images = images.repeat(1, 3, 1, 1)
            # images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            
            images_features = clip_model.encode_image(images)
            images_features = images_features / images_features.norm(dim=-1, keepdim=True)
            CLIP_outputs = images_features @ text_features.T
            cached_clip_logits.append(CLIP_outputs)
            cached_labels.append(labels)
            total_samples += images.size(0)
        cached_clip_logits = torch.cat(cached_clip_logits)
        cached_labels = torch.cat(cached_labels)

    n_batch = math.ceil(total_samples / batch_size)
    best_loss = float('inf')
    best_T = 1.0
    patience = 5
    patience_counter = 0
    ood_probs = ood_probs.to(device)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in range(n_batch):
            start = batch * batch_size
            end = min(total_samples, (batch + 1) * batch_size)
            temp = log_temp.exp()
            source_prob = ood_probs[start:end]
            clip_prob = F.softmax(cached_clip_logits[start:end] / temp, dim=1)
            loss = compute_JS(clip_prob, source_prob)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * (end - start)
        avg_loss = epoch_loss / total_samples
        T = log_temp.exp().item()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_T = T
            patience_counter = 0
        else:
            if abs(T - best_T) <= 0.01:
                patience_counter += 1
            
        scheduler.step(avg_loss)
        print(f"[Epoch {epoch+1}] loss={avg_loss:.4f}, T={log_temp.exp().item():.3f}")

        if patience_counter >= patience:
            return best_T, cached_clip_logits, cached_labels

    T = log_temp.exp().item()
    return T, cached_clip_logits, cached_labels
    
def compute_JS(p, q, eps=1e-8):
    m = 0.5 * (p + q)
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    m = torch.clamp(m, min=eps)
    kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=1)
    kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=1)
    js = 0.5 * (kl_pm + kl_qm)

    return js.mean()

def compute_threshold(temp, dataset, source_model, clip_model, iid_loader, text_features, device):
    source_model.eval()
    # CLIP model loaded to be eval mode
    score_vecs = []
    misclassified = 0
    with torch.no_grad():
        for items in tqdm(iid_loader):
            images, labels = items[0], items[1]
            images, labels = images.to(device), labels.to(device)
            source_outputs = source_model(images)
            source_probs = torch.softmax(source_outputs, dim=1)
            if dataset in ["CIFAR10", "CIFAR100", "MNIST", "TinyImageNet"]:
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            elif dataset in ["ImageNet", "Fmow", "living17", "nonliving26", "entity13", "entity30", "DomainNet"]:
                pass
            # if images.shape[1] == 1:
            #     images = images.repeat(1, 3, 1, 1)
            # images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

            images_features = clip_model.encode_image(images)
            images_features = images_features / images_features.norm(dim=-1, keepdim=True)
            CLIP_outputs = images_features @ text_features.T
            temperatured_CLIP_outputs = CLIP_outputs / temp
            clip_probs = torch.softmax(temperatured_CLIP_outputs, dim=1)
            CLIP_conf, _ = clip_probs.max(dim=1)
            source_conf, source_pred = source_outputs.max(dim=1)
            misclassified += source_pred.ne(labels).sum().item()

            weight = source_conf / (source_conf + CLIP_conf)
            Combine_probs = weight.unsqueeze(1) * source_probs + (1 - weight).unsqueeze(1) * clip_probs
            cur_score = torch.sum(source_probs * Combine_probs, dim=1)
            score_vecs.append(cur_score)
    score_vecs = torch.cat(score_vecs, dim=0)
    t = torch.sort(score_vecs)[0][misclassified-1]
    return t

def get_FRAP_threshold(temp, dataset, net, aid_net, iid_loader, text_features, save_result_path, device):
    if os.path.exists(save_result_path):
        with open(save_result_path, "r") as f:
            data = json.load(f)
            threshold = data["threshold"]
    else:
        os.makedirs(os.path.dirname(save_result_path), exist_ok=True)
        threshold = compute_threshold(temp, dataset, net, aid_net, iid_loader, text_features, device).item()
        with open(save_result_path, "w") as f:
            json.dump({"threshold": threshold}, f)
    return threshold

def get_FRAP(val_temp, test_temp, dataset, source_model, CLIP_model, iid_loader, text_features, ood_probs, cached_clipLogits, save_result_path, device, cached_labels=None):
    threshold = get_FRAP_threshold(val_temp, dataset, source_model, CLIP_model, iid_loader, text_features, save_result_path, device)
    cached_clipLogits = cached_clipLogits.to(device)
    ood_probs = ood_probs.to(device)
    ood_conf, ood_pred = ood_probs.max(dim=1)

    CLIP_probs = F.softmax(cached_clipLogits / test_temp, dim=1)
    CLIP_conf, CLIP_pred = CLIP_probs.max(dim=1)

    weight = ood_conf / (ood_conf + CLIP_conf)
    Combine_probs = weight.unsqueeze(1) * ood_probs + (1 - weight).unsqueeze(1) * CLIP_probs

    estimate_score = torch.sum(ood_probs * Combine_probs, dim=1)
    est = (estimate_score < threshold).sum().item() / len(ood_probs)

    # ------------------------------------------------------------------------------------------------- #
    # Record the data to visualise the calibrate effect 
    # before_clip_conf, _ = F.softmax(cached_clipLogits, dim=1).max(dim=1)
    # if cached_labels is not None:
    #     cached_labels = cached_labels.to(device)
    #     acc = (CLIP_pred.eq(cached_labels).sum().item()) / len(cached_labels)
    # before_calibrate_error = abs(before_clip_conf - acc)
    # after_calibrate_error = abs(CLIP_conf - acc)

    # path_parts = save_result_path.split(os.sep)
    # if len(path_parts) >= 3:
    #     dataset_name = path_parts[-3]
    #     model_name = path_parts[-2]
    #     file_name = path_parts[-1]
    # else:
    #     dataset_name = "unknown_dataset"
    #     model_name = "unknown_model"

    # record_dir = "/home/shuxuan/public_data/Calibrate_CLIP/"
    # os.makedirs(record_dir, exist_ok=True)
    # new_file_name = file_name.replace('threshold', 'calibrate')
    # record_file = os.path.join(record_dir, f"{model_name}_{dataset_name}_{new_file_name}")

    # key = f"{dataset_name}_{model_name}"

    # if os.path.exists(record_file):
    #     with open(record_file, "r") as f:
    #         error_dict = json.load(f)
    # else:
    #     error_dict = {}

    # if key not in error_dict:
    #     error_dict[key] = {
    #         "before_calibrate_error": [],
    #         "after_calibrate_error": []
    #     }
    # error_dict[key]["before_calibrate_error"].append(
    #     float(before_calibrate_error.mean().item()) if hasattr(before_calibrate_error, "mean") else float(before_calibrate_error)
    # )
    # error_dict[key]["after_calibrate_error"].append(
    #     float(after_calibrate_error.mean().item()) if hasattr(after_calibrate_error, "mean") else float(after_calibrate_error)
    # )

    # print("before:", float(before_calibrate_error.mean().item()) if hasattr(before_calibrate_error, "mean") else float(before_calibrate_error))
    # print("after:", float(after_calibrate_error.mean().item()) if hasattr(after_calibrate_error, "mean") else float(after_calibrate_error))

    # with open(record_file, "w") as f:
    #     json.dump(error_dict, f, indent=2)

    # ------------------------------------------------------------------------------------------------- #
    return est

