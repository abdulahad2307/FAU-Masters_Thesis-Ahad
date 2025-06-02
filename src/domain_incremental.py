import argparse
import torch
import os
import time
from torch import nn
from pathlib import Path

from utils.domain_IL.dil_dataloader import DILDataLoader
from utils.domain_IL.dil_model_loader import prepare_dil_models
from utils.domain_IL.dil_train_utils import (
    train_one_epoch_dil,
    evaluate_dil,
    save_checkpoint_dil,
    load_checkpoint_dil,
    set_finetune_mode,
)
from utils.domain_IL.adaptive_lr import AdaptiveLR

def parse_class_counts(class_counts_str, domain_list, default_classes):
    if class_counts_str:
        class_counts = {}
        for item in class_counts_str.split(','):
            domain, count = item.split(':')
            class_counts[domain] = int(count)
        return class_counts
    else:
        return {domain: default_classes for domain in domain_list}

def compute_incremental_learning_gap(current_acc, full_model_acc):
    if full_model_acc is None or full_model_acc >= 1.0:
        return None
    return (current_acc - full_model_acc) / (1 - full_model_acc)

def main(args):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Load class list if provided
    if args.class_list_path:
        import json
        with open(args.class_list_path) as f:
            class_list = json.load(f)
    else:
        class_list = None

    # Data loading
    print("=== Loading Data ===")
    data_start = time.time()
    dil_loader = DILDataLoader(
        data_root=args.data_dir,
        domain_list=args.domain_list,
        batch_size=args.batch_size,
        img_size=(224, 224),
        num_workers=4
    )
    train_loaders = dil_loader.get_domain_loaders('train')
    val_loaders = dil_loader.get_domain_loaders('val')
    test_loaders = dil_loader.get_domain_loaders('test')
    if not train_loaders:
        raise ValueError("No training data found. Check domain paths and directory structure.")
    class_counts = parse_class_counts(args.class_counts, args.domain_list, args.num_classes)
    print(f"Class counts: {class_counts}")
    print(f"Data loaded in {time.time() - data_start:.2f}s")

    # Model preparation
    print("=== Preparing Models ===")
    model_config = {
        'strategy': f'{args.model}_only',
        'eaml_ckpt': args.eaml_path,
        'doc_ckpt': args.docformer_path,
        'dil_mode': True
    }
    if args.ensemble_first and args.model == "both":
        model_config['strategy'] = 'pre_ensemble'
    elif args.model == "both":
        model_config['strategy'] = 'post_ensemble'
    model = prepare_dil_models(model_config, class_counts, device)
    print(f"Setting finetune mode: {args.finetune_mode}")
    if isinstance(model, dict):
        for single_model in model.values():
            set_finetune_mode(single_model, mode=args.finetune_mode, encoder_unfreeze_depth=args.unfreeze_depth)
    else:
        set_finetune_mode(model, mode=args.finetune_mode, encoder_unfreeze_depth=args.unfreeze_depth)

    # Training setup
    print("=== Setting up Training ===")
    if isinstance(model, dict):
        all_params = []
        for single_model in model.values():
            trainable_params = [p for p in single_model.parameters() if p.requires_grad]
            all_params.extend(trainable_params)
    else:
        all_params = [p for p in model.parameters() if p.requires_grad]
    if len(all_params) == 0:
        raise ValueError("No trainable parameters found. Check finetune_mode setting.")
    print(f"Total trainable parameters: {sum(p.numel() for p in all_params):,}")
    optimizer = torch.optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = os.path.join(args.ckpt_dir, f"{args.model}_dil.pth")
    start_epoch = load_checkpoint_dil(model, optimizer, checkpoint_path, device)

    # Adaptive LR scheduler
    lr_sched = AdaptiveLR(optimizer, base_lr=args.lr)

    # Training loop
    print("=== Starting Training ===")
    best_acc = 0.0
    best_model_path = None
    full_model_acc = args.full_model_acc if hasattr(args, "full_model_acc") else None
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch_dil(model, train_loaders, optimizer, criterion, device)
        val_acc = evaluate_dil(model, val_loaders, device)
        g_il = None
        if full_model_acc is not None:
            g_il = ((val_acc*100) - full_model_acc) / (1 - full_model_acc)
        # Adaptive LR step
        val_metrics = {'accuracy': val_acc, 'loss': train_loss}
        lr_adj = lr_sched.step(val_metrics)
        if lr_adj:
            print(f"Learning rate reduced to {lr_sched.get_lr():.3e}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        if g_il is not None:
            print(f"Incremental Learning Gap (G_IL): {g_il:.4f}")
        #print(f"Checkpoint saved: {checkpoint_path}")
        #save_checkpoint_dil(model, optimizer, epoch, checkpoint_path)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = os.path.join(args.ckpt_dir, f"{args.model}_best.pth")
            save_checkpoint_dil(model, optimizer, epoch, best_model_path)
            print(f"New best model saved: {best_acc:.4f}")
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")

    total_time = time.time() - start_time
    print(f"\n=== Training Completed ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Best validation accuracy: {best_acc:.4f}")

    # --- TESTING ON BEST MODEL ---
    if best_model_path is not None and test_loaders is not None:
        print("\n=== Testing Best Model on Test Set ===")
        # Reload best model
        optimizer = torch.optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
        load_checkpoint_dil(model, optimizer, best_model_path, device)
        test_acc = evaluate_dil(model, test_loaders, device)
        g_il_test = None
        if full_model_acc is not None:
            g_il_test = ((test_acc*100) - full_model_acc) / (1 - full_model_acc)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        if g_il_test is not None:
            print(f"Test Incremental Learning Gap (G_IL): {g_il_test:.4f}")
        print("=== End of Test Evaluation ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain Incremental Learning")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory containing domain subdirectories")
    parser.add_argument("--domain_list", nargs="+", required=True, help="List of domain names (subdirectory names)")
    parser.add_argument("--class_counts", type=str, help="Domain-specific class counts (format: domain1:16,domain2:16)")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints_dil")
    parser.add_argument("--eaml_path", type=str, default="")
    parser.add_argument("--docformer_path", type=str, default="")
    parser.add_argument("--model", choices=["eaml", "docformer", "both"], default="eaml")
    parser.add_argument("--ensemble_first", action="store_true", help="Use pre-ensemble (only valid with --model both)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--finetune_mode", choices=["head_only", "partial_finetune", "full_finetune"], default="head_only")
    parser.add_argument("--unfreeze_depth", type=int, default=2)
    parser.add_argument("--class_list_path", type=str, help="Optional path to JSON file with class list")
    parser.add_argument("--num_classes", type=int, default=16, help="Default number of classes per domain")
    parser.add_argument("--full_model_acc", type=float, default=None, help="Full model accuracy for G_IL calculation")
    args = parser.parse_args()
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    main(args)
