import argparse
import os
from data_loader import *
from model_load import get_cv_model, initializa_bert_based_model
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils import get_optimizer, get_lr_scheduler, get_n_classes
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def main():
    parser = argparse.ArgumentParser(description='Train.')
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--data_path', default='/home/shuxuan/public_data/Estimate_Performance', type=str)
    parser.add_argument('--model', default='ResNet18', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--train_epoch', default=20, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)
    parser.add_argument('--save_interval', default=5, type=int)
    parser.add_argument('--resume_epoch', default=0, type=int)

    parser.add_argument('--model_seed', default=1, type=int)

    args = parser.parse_args()

    print(vars(args))

    dataset_name = args.dataset
    data_path = args.data_path
    model_name = args.model
    pretrained = args.pretrained
    n_class = get_n_classes(dataset_name)
    batch_size = args.batch_size
    lr = args.lr
    epoch = args.train_epoch
    resume_epoch = args.resume_epoch
    model_seed = args.model_seed

    if pretrained:
        save_dir = f"/home/shuxuan/public_data/Estimate_Performance/checkpoints/{dataset_name}/{model_name}/pretrained"
    else:
        save_dir = f"/home/shuxuan/public_data/Estimate_Performance/checkpoints/{dataset_name}/{model_name}/scratch"
    
    os.makedirs(save_dir, exist_ok=True)

    if dataset_name == "CIFAR10":
        train_set, train_loader, val_set, val_loader, _, _, _ = get_cifar10(batch_size, data_path, True, True, False)
    elif dataset_name == "CIFAR100":
        train_set, train_loader, val_set, val_loader, _, _, _ = get_cifar100(batch_size, data_path, True, True, False)
    elif dataset_name == "MNIST":
        train_set, train_loader, val_set, val_loader, _, _, _ = get_mnist(batch_size, data_path, True, True, False)
    elif dataset_name == "ImageNet":
        train_set, train_loader, val_set, val_loader, _, _, _ = get_ImageNet(batch_size, data_path, True, True, False)
    elif dataset_name == "TinyImageNet":
        train_set, train_loader, val_set, val_loader, _, _, _ = get_TinyImageNet(batch_size, data_path, True, True, False)
    elif dataset_name in ["living17", "entity13", "entity30", "nonliving26"]:
        train_set, train_loader, val_set, val_loader, _, _, _, label2class = get_imagenet_breeds(batch_size, data_path, dataset_name, True, True, False)
    elif dataset_name == "Camelyon17":
        train_set, train_loader, val_set, val_loader, _, _, _ = get_camelyon17(batch_size, data_path, True, True, False)
    elif dataset_name == "Fmow":
        train_set, train_loader, val_set, val_loader, _, _, _ = get_fmow(batch_size, data_path, True, True, False)
    elif dataset_name == "Rxrx1":
        train_set, train_loader, val_set, val_loader, _, _, _ = get_rxrx1(batch_size, data_path, True, True, False)
    elif dataset_name == "Amazon":
        train_set, train_loader, val_set, val_loader, _, _, _ = get_amazon(batch_size, data_path, 'distilbert-base-uncased', 512, True, True, False)
    elif dataset_name =="CivilComments":
        train_set, train_loader, val_set, val_loader, _, _, _ = get_civilcomments(batch_size, data_path, 'distilbert-base-uncased', 300, True, True, False)
    elif dataset_name =="DomainNet":
        train_set, train_loader, val_set, val_loader, _, _, _ = get_domainnet(batch_size, data_path, True, True, False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'distilbert-base-uncased':
        model = initializa_bert_based_model(n_class)
    else:
        model = get_cv_model(model_name, n_class, model_seed, pretrained)
    
    n_device = torch.cuda.device_count()
    print('available devices:', n_device)
    if n_device > 1:
        model = nn.DataParallel(model, device_ids=range(n_device))
    cudnn.benchmark = False
    model.to(device)

    optimizer = get_optimizer(dataset_name, model, lr, pretrained)
    scheduler = get_lr_scheduler(dataset_name, optimizer, pretrained, T_max=epoch * len(train_loader))

    if resume_epoch > 0:
        ckpt_dir = f"{save_dir}/seed{model_seed}_epoch{resume_epoch}.pt"
        if os.path.exists(ckpt_dir):
            ckpt = torch.load(ckpt_dir, map_location=device)
            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print(f"Checkpoint {ckpt_dir} Loaded.")
        else:
            print(f"Checkpoint {ckpt_dir} not found, starting from scratch.")
    
    print("Start training")
    train(model, optimizer, scheduler, train_loader, val_loader, save_dir, args, device)


def train(net, optimizer, scheduler, train_loader, val_loader, save_dir, args, device):
    net.train()
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(1, args.train_epoch + 1):
        train_loss, correct, total = 0, 0, 0
        start_time = time.time()
        for batch_idx, items in enumerate(train_loader):
            inputs, labels = items[0], items[1]
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 2 == 0:
                current_lr = optimizer.param_groups[0]['lr'] 
                print(f"Epoch: {epoch} ({batch_idx}/{len(train_loader)}) |"
                      f"Loss: {train_loss / (batch_idx + 1):.3f} |"
                      f"ACC: {100. * correct / total:.3f}% ({correct} / {total}) |"
                      f"Lr: {current_lr:.5f}"                    
                )
            if batch_idx % 100 == 0:
                print(f"Time used: {time.time() - start_time:.2f} s")

            if args.dataset == "Rxrx1":
                scheduler.step()
        
        if args.dataset != "Rxrx1":
            scheduler.step()
        
        end_time = time.time()
        print(f"Time used: {end_time - start_time} s")

        if epoch % args.save_interval == 0:
            if isinstance(net, torch.nn.DataParallel):
                model_state_dict = net.module.state_dict()
            else:
                model_state_dict = net.state_dict()
            checkpoint = {
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint, f"{save_dir}/seed{args.model_seed}_epoch{epoch + args.resume_epoch}.pt")
        
        if epoch % args.eval_interval == 0:
            net.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for items in val_loader:
                    inputs, labels = items[0], items[1]
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            net.train()

            print(f'Epoch {epoch} Validation Acc: {val_correct / val_total}')

        if args.resume_epoch + epoch >= args.train_epoch:
            break

    net.eval()
    if isinstance(net, torch.nn.DataParallel):
        model_state_dict = net.module.state_dict()
    else:
        model_state_dict = net.state_dict()
    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, f"{save_dir}/seed{args.model_seed}_epoch{args.train_epoch}.pt")
    print(f"Final Model saved to {save_dir}/seed{args.model_seed}_epoch{args.train_epoch}.pt")

    return net


if __name__ == '__main__':
    main()
            


