import timm
import torch


class AGETimmModel(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
        embedding_size = embedding_size = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)
        self.age_head = torch.nn.Linear(embedding_size, 1)
        self.gender_head = torch.nn.Linear(embedding_size, 1)
        self.ethnicity_head = torch.nn.Linear(embedding_size, 5)
        self.embedding_size = embedding_size

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        embedding = self.backbone.forward_head(feats, pre_logits=True).view(
            -1, self.embedding_size
        )
        return {
            "age": self.age_head(embedding),
            "gender": self.gender_head(embedding),
            "ethnicity": self.ethnicity_head(embedding),
        }


class AGETimmModelAgeCls(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
        embedding_size = embedding_size = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)
        self.age_head = torch.nn.Linear(embedding_size, 9)
        self.gender_head = torch.nn.Linear(embedding_size, 1)
        self.ethnicity_head = torch.nn.Linear(embedding_size, 5)
        self.embedding_size = embedding_size

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        embedding = self.backbone.forward_head(feats, pre_logits=True).view(
            -1, self.embedding_size
        )
        return {
            "age": self.age_head(embedding),
            "gender": self.gender_head(embedding),
            "ethnicity": self.ethnicity_head(embedding),
        }

