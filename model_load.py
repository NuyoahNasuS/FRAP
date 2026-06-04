import torchvision.models as models
from transformers import DistilBertForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)  # 64*12*12 = 9216
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def get_cv_model_base(model, num_classes, pretrained=True):
    target_model = None
    if model == "DenseNet121":
        target_model = models.densenet121(pretrained=pretrained)
        target_model.classifier = nn.Linear(target_model.classifier.in_features, num_classes)
    elif model == "ResNet18":
        target_model = models.resnet18(pretrained=pretrained)
        target_model.fc = nn.Linear(target_model.fc.in_features, num_classes)
    elif model == "ResNet50":
        target_model = models.resnet50(pretrained=pretrained)
        target_model.fc = nn.Linear(target_model.fc.in_features, num_classes)
    elif model == "VGG11":
        target_model = models.vgg11(pretrained=pretrained)
        target_model.classifier[6] = nn.Linear(target_model.classifier[6].in_features, num_classes)
    elif model == "ViT_B_16":
        target_model = models.vit_b_16(pretrained=pretrained)
        target_model.heads.head = nn.Linear(target_model.heads.head.in_features, num_classes)
    elif model == "SimpleConv":
        target_model = SimpleConv()
    else:
        raise ValueError(f"{model} not considered")
    
    return target_model

def get_cv_model(model, num_classes, seed=42, pretrained=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    target_model = get_cv_model_base(model=model, num_classes=num_classes, pretrained=pretrained)
    return target_model

class DistilBertClassifier(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        return outputs
    
def initializa_bert_based_model(num_classes):
    model = DistilBertClassifier.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_classes
    )
    return model

