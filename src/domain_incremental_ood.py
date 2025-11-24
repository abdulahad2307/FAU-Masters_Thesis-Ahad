import os
import random
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from torch.utils.data import Dataset, DataLoader

from utils.domain_IL.dil_dataloader import DILDataLoader, safe_collate
from utils.domain_IL.dil_train_utils import (
    save_checkpoint_dil, save_epoch_checkpoint_dil, train_one_epoch_dil,
    evaluate_dil, classwise_accuracy, evaluate_domain
)
from utils.domain_IL.dil_utils import (
    StandardDomainIL, DistillationDomainIL, EWC, ExemplarManager, AdaptiveLR
)
from utils.domain_IL.dil_model_loader import load_eaml_model_partial, set_finetune_mode

from utils.ood.msp import MSP_OOD
from utils.ood.vim import VIM_OOD
from utils.ood.gradnorm import GradNorm_OOD
from utils.ood.ood_viz import plot_ood_histograms
from utils.domain_IL.dil_utils import extract_features, extract_features_and_logits

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FilteredDataset(Dataset):
    def __init__(self, base_dataset, target_class_idx):
        self.base_dataset = base_dataset
        self.class_idx = target_class_idx
        self.sample_indices = [i for i, (_, lbl) in enumerate(base_dataset.samples) if lbl == target_class_idx]
        self.transform = getattr(base_dataset, 'transform', None)
        self.collate_fn = getattr(base_dataset, 'collate_fn', None)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        actual_idx = self.sample_indices[idx]
        return self.base_dataset[actual_idx]

def run_domain_incremental_ood(
    data_root,
    ocr_tensor_dirs,
    domains,
    global_classes,
    eaml_ckpt_path,
    checkpoint_dir,
    ood_method="vim",
    batch_size=16,
    lr=1e-4,
    weight_decay=1e-4,
    num_epochs=10,
    strategy="distillation",
    temperature=2.0,
    lambda_distill=1.0,
    lambda_ewc=5000.0,
    use_ewc=True,
    use_exemplars=False,
    max_exemplars=200,
    exemplar_selection="random",
    finetune_mode="head_only",
    unfreeze_depth=0,
    patience=10,
    use_bias_correction=True,
    resume=False,
    resume_ckpt_path=None,
    ood_threshold=0.75
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    class_to_idx = {cls: idx for idx, cls in enumerate(global_classes)}

    inc_strategy = DistillationDomainIL(DEVICE, temperature, lambda_distill) \
        if strategy == "distillation" else StandardDomainIL(DEVICE)

    exemplar_mgr = ExemplarManager(max_exemplars=max_exemplars, selection_strategy=exemplar_selection) \
        if use_exemplars else None

    dil_loader = DILDataLoader(
        data_root=data_root,
        domain_list=domains,
        batch_size=batch_size,
        img_size=(229, 229),
        num_workers=4,
        ocr_tensor_dirs=ocr_tensor_dirs,
        class_to_idx=class_to_idx
    )

    pretrained_domain = domains[0]
    incremental_domain = domains[1]

    loaders_train = dil_loader.get_domain_loaders('train')
    loaders_val = dil_loader.get_domain_loaders('val')
    loaders_test = dil_loader.get_domain_loaders('test')

    train_loader = loaders_train.get(incremental_domain)
    val_loader = loaders_val.get(incremental_domain)
    test_loader_pretrained = loaders_test.get(pretrained_domain)
    test_loader_incremental = loaders_test.get(incremental_domain)

    ood_methods = {
        "msp": MSP_OOD,
        "vim": VIM_OOD,
        "gradnorm": GradNorm_OOD
    }
    assert ood_method in ood_methods, f"OOD method '{ood_method}' not implemented."
    OODClass = ood_methods[ood_method]
    ood_detector = OODClass(threshold=ood_threshold)

    old_classes = 16
    new_classes = len(global_classes)

    model = load_eaml_model_partial(
        eaml_ckpt_path, old_classes, new_classes,
        device=DEVICE, text_branch=True
    )
    model = set_finetune_mode(model, finetune_mode, unfreeze_depth)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, weight_decay=weight_decay)
    lr_scheduler = AdaptiveLR(optimizer)
    criterion = torch.nn.CrossEntropyLoss()

    ewc = EWC(model, test_loader_pretrained, DEVICE, lambda_ewc) if use_ewc else None
    old_model = load_eaml_model_partial(eaml_ckpt_path, old_classes, new_classes,
                                        device=DEVICE, text_branch=True)
    old_model.eval()

    best_val_acc, best_val_loss, best_model_path = 0.0, float('inf'), None
    no_improve, start_epoch = 0, 1

    if resume and resume_ckpt_path and os.path.exists(resume_ckpt_path):
        print(f"Resuming from checkpoint: {resume_ckpt_path}")
        ckpt = torch.load(resume_ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        no_improve = ckpt.get('no_improve', 0)
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        del ckpt
        torch.cuda.empty_cache()

    print("=== Domain Incremental Training with OOD ===")
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loader_dict = {incremental_domain: train_loader}
        train_loss, train_acc = train_one_epoch_dil(
            model=model, train_loaders=train_loader_dict,
            optimizer=optimizer, criterion=criterion,
            device=DEVICE, strategy=inc_strategy,
            exemplar_manager=exemplar_mgr, ewc=ewc,
            old_model=old_model if isinstance(inc_strategy, DistillationDomainIL) else None,
            use_bias_correction=use_bias_correction
        )

        val_loss, val_acc = evaluate_dil(model, {incremental_domain: val_loader}, DEVICE)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        improved = val_acc > best_val_acc or val_loss < best_val_loss
        if improved:
            best_val_acc, best_val_loss = val_acc, val_loss
            no_improve = 0
        else:
            no_improve += 1
            print(f"Patience {no_improve}/{patience}")
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
        save_checkpoint_dil(model, optimizer, epoch, checkpoint_path,
                            extra_data={"best_val_acc": best_val_acc, "best_val_loss": best_val_loss})
        if improved:
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            save_checkpoint_dil(model, optimizer, epoch, best_model_path)
            print("Validation improved — evaluating OOD...")

            if ood_method == "vim":
                features_by_class = {}
                logits_by_class = {}

                base_dataset = dil_loader.get_full_dataset(incremental_domain)
                if base_dataset is None:
                    raise RuntimeError(f"Failed to obtain dataset for domain '{incremental_domain}'")

                for cls in global_classes:
                    cls_idx = class_to_idx[cls]
                    filtered_ds = FilteredDataset(base_dataset, cls_idx)
                    if len(filtered_ds) == 0:
                        print(f"Warning: No samples for class '{cls}', skipping feature extraction.")
                        features_by_class[cls] = None
                        logits_by_class[cls] = None
                        continue
                    class_loader = DataLoader(
                        filtered_ds,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4,
                        collate_fn=safe_collate
                    )
                    cls_feats, cls_logits = extract_features_and_logits(model, class_loader, DEVICE)
                    features_by_class[cls] = cls_feats
                    logits_by_class[cls] = cls_logits

                # Filter out classes with None features/logits
                filtered_features = {cls: feats for cls, feats in features_by_class.items() if feats is not None}
                filtered_logits = {cls: logs for cls, logs in logits_by_class.items() if logs is not None}

                #print("DEBUG: Features by class:")
                for cls, feats in filtered_features.items():
                    feat_type = type(feats)
                    if hasattr(feats, '__len__'):
                        example_type = type(feats[0]) if len(feats) > 0 else 'N/A'
                        shape = feats.shape if hasattr(feats, 'shape') else 'N/A'
                    else:
                        example_type = 'N/A'
                        shape = 'N/A'
                    #print(f"  Class {cls}: {feat_type}, example element type: {example_type}, shape: {shape}")

                #print("DEBUG: Logits by class:")
                for cls, logs in filtered_logits.items():
                    log_type = type(logs)
                    if hasattr(logs, '__len__'):
                        example_type = type(logs[0]) if len(logs) > 0 else 'N/A'
                        shape = logs.shape if hasattr(logs, 'shape') else 'N/A'
                    else:
                        example_type = 'N/A'
                        shape = 'N/A'
                    #print(f"  Class {cls}: {log_type}, example element type: {example_type}, shape: {shape}")

                ood_detector.fit(filtered_features, filtered_logits)
                val_features_dict = filtered_features
                ood_result = ood_detector.ood_metrics(val_features_dict, known_classes=global_classes)


            print(f"[OOD-{ood_method.upper()}] Val Open-Set Acc: {ood_result['open_set_accuracy']:.4f}, "
                  f"Unknown Reject: {ood_result['unknown_rejection']:.4f}")

            lr_scheduler.step({"accuracy": val_acc, "loss": val_loss})

    print("\n=== Final Model Evaluation and OOD Testing ===")
    final_model = model
    test_sets = {
        "Pretrained": test_loader_pretrained,
        "Incremental": test_loader_incremental
    }

    for name, loader in test_sets.items():
        acc = evaluate_domain(final_model, loader, DEVICE, len(global_classes), global_classes)
        print(f"Test Accuracy on {name} Domain: {acc:.4f}")

        feats = extract_features(final_model, loader, DEVICE)
        ood_scores = ood_detector.ood_metrics(feats, known_classes=global_classes)
        print(f"[OOD-{ood_method.upper()}] {name} - Open-Set Acc: {ood_scores['open_set_accuracy']:.4f}, "
              f"Unknown Reject: {ood_scores['unknown_rejection']:.4f}")

        savepath = os.path.join(checkpoint_dir, f"ood_{ood_method}_{name.lower()}.png")
        if ood_scores.get("y_true") is not None and ood_scores.get("scores") is not None:
            plot_ood_histograms(
                ood_scores["y_true"], ood_scores["y_pred"], ood_scores["scores"], savepath
            )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--ocr_tensor_dirs', nargs=2, required=True)
    p.add_argument('--domains', required=True)
    p.add_argument('--global_classes', required=True)
    p.add_argument('--eaml_ckpt_path', required=True)
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--ood_method', type=str, default='msp', choices=['msp', 'vim', 'gradnorm'])
    p.add_argument('--ood_threshold', type=float, default=0.75)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num_epochs', type=int, default=10)
    p.add_argument('--strategy', choices=['standard','distillation'], default='distillation')
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--lambda_distill', type=float, default=1.0)
    p.add_argument('--lambda_ewc', type=float, default=5000.0)
    p.add_argument('--use_ewc', action='store_true')
    p.add_argument('--use_exemplars', action='store_true')
    p.add_argument('--max_exemplars', type=int, default=200)
    p.add_argument('--exemplar_selection', choices=['random','herding'], default='random')
    p.add_argument('--finetune_mode', choices=['full_finetune', 'head_only', 'partial_finetune'], default='head_only')
    p.add_argument('--unfreeze_depth', type=int, default=0)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--use_bias_correction', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--resume_ckpt_path', type=str)
    args = p.parse_args()

    run_domain_incremental_ood(
        data_root=args.data_dir,
        ocr_tensor_dirs={d: t for d, t in zip(args.domains.split(','), args.ocr_tensor_dirs)},
        domains=args.domains.split(','),
        global_classes=args.global_classes.split(','),
        eaml_ckpt_path=args.eaml_ckpt_path,
        checkpoint_dir=args.checkpoint_dir,
        ood_method=args.ood_method,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        strategy=args.strategy,
        temperature=args.temperature,
        lambda_distill=args.lambda_distill,
        lambda_ewc=args.lambda_ewc,
        use_ewc=args.use_ewc,
        use_exemplars=args.use_exemplars,
        max_exemplars=args.max_exemplars,
        exemplar_selection=args.exemplar_selection,
        finetune_mode=args.finetune_mode,
        unfreeze_depth=args.unfreeze_depth,
        patience=args.patience,
        use_bias_correction=args.use_bias_correction,
        resume=args.resume,
        resume_ckpt_path=args.resume_ckpt_path,
        ood_threshold=args.ood_threshold
    )
