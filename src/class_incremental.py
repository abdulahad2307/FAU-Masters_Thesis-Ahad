import os
import time
import torch
import torch.nn as nn
from typing import List, Optional

from utils.eaml.eaml_model import EAMLModel
from utils.docformer.model import DocFormer
from utils.docformer.config import DocFormerConfig
from utils.class_IL.dataloader_utils import get_class_il_loader
from utils.class_IL.train_utils import (
    save_checkpoint, load_checkpoint, evaluate, CILMetrics
)
from utils.class_IL.cil_utils import (
    StandardIncremental, DistillationIncremental,
    EWC, ExemplarManager, AdaptiveLR, extract_features
)
from utils.class_IL.training_modes import get_training_mode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_incremental_learning(
    data_root: str,
    class_order: List[str],
    base_model_path: str,
    model_name: str,
    checkpoint_dir: str,
    start_step: int = 0,
    batch_size: int = 8,
    lr: float = 2e-5,
    num_epochs: int = 10,
    strategy: str = "distillation",
    temperature: float = 2.0,
    lambda_distill: float = 1.0,
    lambda_ewc: float = 5000.0,
    use_ewc: bool = False,
    use_exemplars: bool = False,
    max_exemplars: int = 200,
    exemplar_selection: str = "herding",
    training_mode: str = "last_layer",
    trainable_layers: Optional[List[str]] = None,
    resume_checkpoint: Optional[str] = None
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("Starting Class Incremental Learning...")

    # Metrics tracker
    initial_classes = class_order[: start_step + 1] if start_step > 0 else [class_order[0]]
    metrics = CILMetrics(initial_classes)

    # Choose incremental strategy
    if strategy == "distillation":
        inc_strategy = DistillationIncremental(DEVICE, temperature, lambda_distill)
    else:
        inc_strategy = StandardIncremental(DEVICE)

    # Exemplar manager
    exemplar_mgr = ExemplarManager(
        max_exemplars=max_exemplars,
        selection_strategy=exemplar_selection
    ) if use_exemplars else None

    best_acc = 0.0
    old_model = None
    ewc = None

    for step in range(start_step, len(class_order)):
        current = class_order[: step + 1]
        previous = class_order[:step]
        new_cls = [class_order[step]] if step > start_step else []

        print(f"\n=== Step {step+1}/{len(class_order)} - Classes: {current} ===")
        if step > start_step:
            metrics.incremental_state_update(new_cls)

        # Data loaders
        train_loader = get_class_il_loader(model_name, os.path.join(data_root, "train"), current, batch_size)
        val_loader   = get_class_il_loader(model_name, os.path.join(data_root, "val"),   current, batch_size)

        # Build model
        if model_name == "docformer":
            cfg = DocFormerConfig()
            nc = len(train_loader.dataset.class_to_idx)
            model = DocFormer(cfg, num_classes=nc).to(cfg.device)
        else:  # EAML
            nc = len(initial_classes) if step == start_step else len(current)
            model = EAMLModel(num_classes=nc).to(DEVICE)

        # Load base checkpoint at first step
        if step == start_step and os.path.exists(base_model_path):
            checkpoint = torch.load(base_model_path, map_location=DEVICE)
            state = checkpoint.get("model_state_dict", checkpoint)
            # avoid classifier mismatch
            own = model.state_dict()
            for k, v in state.items():
                if k in own and v.size() == own[k].size():
                    own[k] = v
            model.load_state_dict(own)
            print(f"Loaded base model from {base_model_path}")

        # Prepare old_model & EWC after first step
        if step > start_step:
            # Save old for distillation and EWC
            old_model = EAMLModel(num_classes=len(previous)).to(DEVICE) \
                if model_name == "eaml" else DocFormer(cfg, num_classes=len(previous)).to(cfg.device)
            prev_ckpt = os.path.join(checkpoint_dir, f"step_{step-1}_class_{previous[-1]}.pth")
            if os.path.exists(prev_ckpt):
                prev = torch.load(prev_ckpt, map_location=DEVICE)
                old_model.load_state_dict(prev.get("model_state_dict", prev))
            old_model.eval()

            # Adapt classifier for new classes
            model = inc_strategy.adapt_model(model, len(previous), len(current), model_name)

        # Training mode (full / last_layer / selective)
        tm = get_training_mode(model, training_mode, trainable_layers)
        model = tm.prepare_for_training()
        optimizer = torch.optim.AdamW(tm.get_trainable_params(), lr=lr)

        # Resume from checkpoint if requested
        start_epoch = 0
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            start_epoch = load_checkpoint(model, optimizer, resume_checkpoint, DEVICE)

        # EWC initialization
        if use_ewc and step > start_step:
            exemplar_loader = train_loader if not exemplar_mgr else \
                get_class_il_loader(model_name, os.path.join(data_root, "train"), previous, batch_size)
            ewc = EWC(old_model, exemplar_loader, DEVICE, lambda_ewc)

        criterion = nn.CrossEntropyLoss()
        lr_sched = AdaptiveLR(optimizer, base_lr=lr)

        # Main training loop
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                loss, preds, labels = inc_strategy.compute_loss(model, batch, criterion, old_model, ewc)
                loss.backward()
                optimizer.step()
                metrics.update(preds.cpu().numpy(), labels.cpu().numpy())
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

            # Validation
            val_metrics = evaluate(model, val_loader, DEVICE, metrics)
            lr_adj = lr_sched.step(val_metrics['top1_acc'])
            if lr_adj:
                print(f"Learning rate reduced to {lr_sched.get_lr():.3e}")

            if val_metrics['top1_acc'] > best_acc:
                best_acc = val_metrics['top1_acc']
                save_checkpoint(model, optimizer, epoch+1,
                                os.path.join(checkpoint_dir, f"step_{step}_class_{class_order[step]}.pth"))

        # Exemplar update
        if exemplar_mgr:
            exemplar_mgr.set_feature_extractor(model)
            exemplar_mgr.update(train_loader.dataset, class_order[step], model)

    # Save final model
    save_checkpoint(model, optimizer, num_epochs,
                    os.path.join(checkpoint_dir, "final_model.pth"))
    print("Incremental Learning completed successfully!")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--class_order', required=True,
                   help="Comma-separated class order")
    p.add_argument('--base_model_path', required=True)
    p.add_argument('--model_name', required=True, choices=['eaml', 'docformer'])
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--start_step', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--num_epochs', type=int, default=10)
    p.add_argument('--strategy', choices=['standard','distillation'], default='distillation')
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--lambda_distill', type=float, default=1.0)
    p.add_argument('--use_ewc', action='store_true')
    p.add_argument('--lambda_ewc', type=float, default=5000.0)
    p.add_argument('--use_exemplars', action='store_true')
    p.add_argument('--max_exemplars', type=int, default=200)
    p.add_argument('--exemplar_selection', choices=['random','herding'], default='herding')
    p.add_argument('--training_mode', choices=['full','last_layer','selective'], default='last_layer')
    p.add_argument('--trainable_layers', nargs='+', default=None)
    p.add_argument('--resume_checkpoint', type=str, default=None,
                   help="Path to checkpoint for resuming training")

    args = p.parse_args()
    run_incremental_learning(
        data_root=args.data_dir,
        class_order=args.class_order.split(','),
        base_model_path=args.base_model_path,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        start_step=args.start_step,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        strategy=args.strategy,
        temperature=args.temperature,
        lambda_distill=args.lambda_distill,
        use_ewc=args.use_ewc,
        lambda_ewc=args.lambda_ewc,
        use_exemplars=args.use_exemplars,
        max_exemplars=args.max_exemplars,
        exemplar_selection=args.exemplar_selection,
        training_mode=args.training_mode,
        trainable_layers=args.trainable_layers,
        resume_checkpoint=args.resume_checkpoint
    )
