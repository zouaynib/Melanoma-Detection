# src/train/lightning_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

import torchmetrics
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
    MulticlassAUROC,
)

from models.combined_model import CombinedModel


class ISICLightningModule(LightningModule):
    def __init__(
        self,
        backbone_name: str,
        metadata_dim: int,
        num_classes: int = 8,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.model = CombinedModel(
            backbone_name=backbone_name,
            pretrained_backbone=True,
            metadata_dim=metadata_dim,
            num_classes=num_classes,
        )

        # Warmup threshold
        self.warmup_epochs = warmup_epochs

        # Loss
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        # Freeze backbone initially
        self.freeze_backbone = True
        self._apply_backbone_freeze()

        # ----------------------------
        # TorchMetrics setup
        # ----------------------------
        self.num_classes = num_classes

        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_f1_weighted = MulticlassF1Score(num_classes=num_classes, average="weighted")

        self.val_precision = MulticlassPrecision(num_classes=num_classes, average=None)
        self.val_recall = MulticlassRecall(num_classes=num_classes, average=None)
        self.val_f1_perclass = MulticlassF1Score(num_classes=num_classes, average=None)

        self.val_confmat = MulticlassConfusionMatrix(num_classes=num_classes)
        self.val_auroc = MulticlassAUROC(num_classes=num_classes)

        # To store predictions for confusion matrix / ROC
        self.val_preds = []
        self.val_labels = []

    # -------------------------------------------------------
    # Backbone freeze/unfreeze utilities
    # -------------------------------------------------------
    def _apply_backbone_freeze(self):
        for p in self.model.backbone.parameters():
            p.requires_grad = False

    def _apply_backbone_unfreeze(self):
        for p in self.model.backbone.parameters():
            p.requires_grad = True

    # -------------------------------------------------------
    # Forward
    # -------------------------------------------------------
    def forward(self, images, metadata):
        return self.model(images, metadata)

    # -------------------------------------------------------
    # Generic step
    # -------------------------------------------------------
    def _step(self, batch, stage: str):
        images, meta, labels = batch
        logits = self(images, meta)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)


        return loss, logits, labels

    # -------------------------------------------------------
    # Training
    # -------------------------------------------------------
    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, stage="train")
        return loss

    # -------------------------------------------------------
    # Validation
    # -------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._step(batch, stage="val")

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        # Store for confusion matrix / ROC
        self.val_preds.append(probs.detach().cpu())
        self.val_labels.append(labels.detach().cpu())

        # TorchMetrics updates
        self.val_acc.update(preds, labels)
        self.val_f1_macro.update(preds, labels)
        self.val_f1_weighted.update(preds, labels)

        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_f1_perclass.update(preds, labels)

        self.val_confmat.update(preds, labels)
        self.val_auroc.update(probs, labels)

    # Called once per epoch after all val batches
    def on_validation_epoch_end(self):
        # Compute metrics
        acc = self.val_acc.compute()
        f1_macro = self.val_f1_macro.compute()
        f1_weighted = self.val_f1_weighted.compute()
        prec = self.val_precision.compute()
        rec = self.val_recall.compute()
        f1_pc = self.val_f1_perclass.compute()
        confmat = self.val_confmat.compute()
        auroc = self.val_auroc.compute()

        # Log summary metrics
        self.log("val_macro_f1", f1_macro, sync_dist=True)
        self.log("val_weighted_f1", f1_weighted, sync_dist=True)
        self.log("val_auroc", auroc, sync_dist=True)


        # Log per-class metrics
        for i in range(self.num_classes):
            self.log(f"val_precision_class_{i}", prec[i], sync_dist=True)
            self.log(f"val_recall_class_{i}", rec[i], sync_dist=True)
            self.log(f"val_f1_class_{i}", f1_pc[i], sync_dist=True)


        # ---------------------------------------------------------
        # Confusion matrix plot (manual Matplotlib version)
        # ---------------------------------------------------------
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            confmat.cpu().numpy(),
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        self.logger.experiment.add_figure(
            "confusion_matrix",
            fig,
            global_step=self.current_epoch,
        )
        plt.close(fig)

        # Reset state
        self.val_acc.reset()
        self.val_f1_macro.reset()
        self.val_f1_weighted.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1_perclass.reset()
        self.val_confmat.reset()
        self.val_auroc.reset()

        self.val_preds.clear()
        self.val_labels.clear()


    # -------------------------------------------------------
    # Optimizers
    # -------------------------------------------------------
    def configure_optimizers(self):
        backbone_lr = self.hparams.lr * 0.1   # 1e-5 if lr=1e-4
        head_lr = self.hparams.lr             # 1e-4

        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.backbone.parameters(), "lr": backbone_lr},
                {"params": self.model.meta_mlp.parameters(), "lr": head_lr},
                {"params": self.model.classifier.parameters(), "lr": head_lr},
            ],
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
