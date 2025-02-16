import torch
import torch.nn as nn
import torchvision.models as models

class PerceptAnonHA1(nn.Module):
    '''
        PerceptAnon HA1 Network for HA1 (single anonymized image)
    '''
    def __init__(self, num_outputs=None, pretrained=True, is_classification=True):
        super(PerceptAnonHA1, self).__init__()
        self.num_outputs = num_outputs
        self.pretrained = pretrained
        self.is_classification = is_classification

    def _get_base_model(self, model_name):
        # Get pytorch model architectures with pretrained weights if selected
        if model_name == 'resnet18':
            return models.resnet18(weights=models.ResNet18_Weights.DEFAULT) if self.pretrained else models.resnet18()
        elif model_name == 'resnet50':
            return models.resnet50(weights=models.ResNet50_Weights.DEFAULT) if self.pretrained else models.resnet50()
        elif model_name == 'resnet152':
            return models.resnet152(weights=models.ResNet152_Weights.DEFAULT) if self.pretrained else models.resnet152()
        elif model_name == 'densenet121':
            return models.densenet121(weights=models.DenseNet121_Weights.DEFAULT) if self.pretrained else models.densenet121()
        elif model_name == 'vgg11':
            return models.vgg11(weights=models.VGG11_Weights.DEFAULT) if self.pretrained else models.vgg11()
        elif model_name == 'alexnet':
            return models.alexnet(weights=models.AlexNet_Weights.DEFAULT) if self.pretrained else models.alexnet()
        elif model_name == 'vit_b_16':
            return models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT) if self.pretrained else models.vit_b_16()
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def _modify_model(self, model_name, model):
        # ResNets
        if 'resnet' in model_name: #and hasattr(model, 'fc'):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_outputs)
        # DenseNets
        elif 'densenet' in model_name: #hasattr(model, 'classifier'):
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, self.num_outputs)
        # VGGs or AlexNet
        elif 'vgg' in model_name or 'alexnet' in model_name: #isinstance(model.classifier, nn.Sequential):
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, self.num_outputs)
        # ViT
        elif 'vit' in model_name:
            num_ftrs = model.heads.head.in_features
            model.heads.head = nn.Linear(num_ftrs, self.num_outputs)
        
        # Add Sigmoid at end for regression 
        if not self.is_classification and self.num_outputs == 1:
            original_forward = model.forward
            def forward_with_sigmoid(*args, **kwargs):
                x = original_forward(*args, **kwargs)
                return torch.sigmoid(x)
            model.forward = forward_with_sigmoid
        
        return model
    
    def _modify_model_for_embedding(self, model):
        # Remove the final classification layer to get the embeddings
        if isinstance(model, models.ResNet) or isinstance(model, models.DenseNet):
            # https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
            model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        elif isinstance(model, models.ViT):
            model.heads.head = nn.Identity()
            model = nn.Sequential(model, nn.Flatten())
        else:
            raise ValueError(f"Unsupported model architecture: {type(model)}")

        return model

    def get_model(self, model_name):
        model = self._get_base_model(model_name)
        model = self._modify_model(model_name, model)
        return model

    def get_feature_extractor(self, model_name):
        model = self._get_base_model(model_name)
        model = self._modify_model_for_embedding(model)
        return model


class PerceptAnonHA2(nn.Module):
    '''
        PerceptAnon Siamese Network for HA2 (original-anonymized image pairs)
    '''
    def __init__(self, model_name, num_outputs, pretrained=True, is_classification=True):
        super(PerceptAnonHA2, self).__init__()
        self.num_outputs = num_outputs
        self.encoder_feature_dim = None
        self.pretrained = pretrained
        self.is_classification = is_classification
        base_model = self._get_base_model(model_name)
        self.shared_extractor = self._modify_model_for_embedding(base_model)        

        # Regression or classification head
        self.head = nn.Sequential(
            nn.Linear(2 * self.encoder_feature_dim, 512),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(512, self.num_outputs)
        )

        if not self.is_classification and self.num_outputs == 1:
            original_forward = self.forward
            def forward_with_sigmoid(*args, **kwargs):
                x = original_forward(*args, **kwargs)
                return torch.sigmoid(x)
            self.forward = forward_with_sigmoid
        
    def forward(self, img1, img2):
        # Get embeddings
        emb1 = self.shared_extractor(img1)
        emb2 = self.shared_extractor(img2)
        # Concatenate features
        combined = torch.cat((emb1, emb2), dim=1)
        # Pass through the regression or classification head
        output = self.head(combined)
        return output

    def _get_base_model(self, model_name):
        if model_name == 'resnet18':
            return models.resnet18(weights=models.ResNet18_Weights.DEFAULT) if self.pretrained else models.resnet18()
        elif model_name == 'resnet50':
            return models.resnet50(weights=models.ResNet50_Weights.DEFAULT) if self.pretrained else models.resnet50()
        elif model_name == 'resnet152':
            return models.resnet152(weights=models.ResNet152_Weights.DEFAULT) if self.pretrained else models.resnet152()
        elif model_name == 'densenet121':
            return models.densenet121(weights=models.DenseNet121_Weights.DEFAULT) if self.pretrained else models.densenet121()
        elif model_name == 'vgg11':
            return models.vgg11(weights=models.VGG11_Weights.DEFAULT) if self.pretrained else models.vgg11()
        elif model_name == 'alexnet':
            return models.alexnet(weights=models.AlexNet_Weights.DEFAULT) if self.pretrained else models.alexnet()
        elif model_name == 'vit_b_16':
            return models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT) if self.pretrained else models.vit_b_16()
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
    def _modify_model_for_embedding(self, model):
        # Remove the final classification layer to get the embeddings
        # https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/5
        # https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
        if isinstance(model, models.ResNet):
            # model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
            self.encoder_feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif isinstance(model, models.DenseNet):
            self.encoder_feature_dim = model.classifier.in_features
            model.classifier = nn.Identity()
        elif isinstance(model, models.VGG) or isinstance(model, models.AlexNet):
            self.encoder_feature_dim = model.classifier[6].in_features
            model.classifier[6] = nn.Identity()
        elif isinstance(model, models.ViT):
            model.heads.head = nn.Identity()
            # model = nn.Sequential(model, nn.Flatten())
        else:
            raise ValueError(f"Unsupported model architecture: {type(model)}")

        return model

# ----------------------------------

class RegHead(nn.Module):
    def __init__(self, feature_dim):
        super(RegHead, self).__init__()
        self.regression = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.regression(x))
    
class ClfHead(nn.Module):
    def __init__(self, feature_dim, num_outputs):
        super(ClfHead, self).__init__()
        self.regression = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(512, num_outputs)
        )

    def forward(self, x):
        return self.regression(x)