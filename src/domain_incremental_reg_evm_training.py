import os
import random
import torch
import torch.nn as nn
import numpy as np
from utils.domain_IL.dil_dataloader import DILDataLoader
from utils.domain_IL.dil_train_utils import (
    save_checkpoint_dil, save_epoch_checkpoint_dil, train_one_epoch_dil,
    evaluate_dil, classwise_accuracy, evaluate_domain
)
from utils.domain_IL.dil_utils import (
    StandardDomainIL, DistillationDomainIL, EWC, ExemplarManager, AdaptiveLR
)
from utils.domain_IL.dil_model_loader import load_eaml_model_partial, set_finetune_mode
from utils.evm.evm_classifier_reg import RegularizedEVMClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features_for_evm(model, loader, device, global_classes):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            texts = batch.get("texts", None)
            if texts is not None:
                texts = {k: v.to(device) for k, v in texts.items()}
                feats = model.extract_features(imgs, texts)
            else:
                feats = model.extract_features(imgs)
            features.append(feats.cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    feature_dict = {}
    for i, l in enumerate(labels):
        class_name = global_classes[l] if isinstance(global_classes[0], str) else l
        feature_dict.setdefault(class_name, []).append(features[i])
    for k in feature_dict:
        feature_dict[k] = np.stack(feature_dict[k], axis=0)
    return feature_dict, features, labels

def hybrid_loss(logits, features, labels, evm, criterion, global_classes, lambda_evm):
    ce_loss = criterion(logits, labels)
    features_np = features.detach().cpu().numpy()
    label_strs = [global_classes[l] if isinstance(global_classes[0], str) else l for l in labels.cpu().numpy()]
    evm_probs = evm.predict_proba(features_np)
    prob_matrix = np.stack([evm_probs[c] for c in sorted(evm_probs.keys())], axis=1)   # (n_samples, n_classes)
    sample_indices = [sorted(evm_probs.keys()).index(lstr) for lstr in label_strs]
    prob_true = prob_matrix[np.arange(len(label_strs)), sample_indices]
    evm_loss = -np.log(prob_true + 1e-6).mean()  # negative log-likelihood known
    evm_loss_tensor = torch.tensor(evm_loss, dtype=torch.float32, device=logits.device)
    return ce_loss + lambda_evm * evm_loss_tensor

def train_one_epoch_dil_evm(
    model,
    train_loaders,
    optimizer,
    criterion,
    device,
    strategy,
    global_classes,
    ewc=None,
    exemplar_manager=None,
    old_model=None,
    use_bias_correction=False,
    evm=None,
    lambda_evm=0.0
):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    batch_count = 0
    total_batches = sum(len(loader) for loader in train_loaders.values())
    for domain, loader in train_loaders.items():
        batch_features, batch_labels = [], []
        for batch in loader:
            inputs = batch["images"].to(device)
            labels = batch["labels"].to(device)
            text_inputs = batch.get("texts", None)
            if text_inputs is not None:
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                feats = model.extract_features(inputs, text_inputs)
            else:
                feats = model.extract_features(inputs)
            batch_features.append(feats.cpu())
            batch_labels.append(labels.cpu())
            batch_data = {"images": inputs, "labels": labels}
            if text_inputs is not None:
                batch_data["texts"] = text_inputs
            logits = model(inputs, text_inputs) if text_inputs is not None else model(inputs)
            loss = hybrid_loss(logits, feats, labels, evm, criterion, global_classes, lambda_evm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += (logits.argmax(1) == labels).float().mean().item()
            batch_count += 1
        features_epoch = torch.cat(batch_features, 0)
        labels_epoch = torch.cat(batch_labels, 0)
        # Update EVM with epoch features (incremental fitting)
        feature_dict = {}
        label_vals = labels_epoch.numpy()
        for i, l in enumerate(label_vals):
            class_name = global_classes[l] if isinstance(global_classes[0], str) else l
            feature_dict.setdefault(class_name, []).append(features_epoch[i].numpy())
        for k in feature_dict:
            feature_dict[k] = np.stack(feature_dict[k], axis=0)
        if evm is not None:
            evm.incremental_update(feature_dict)
    if use_bias_correction and hasattr(model, "fusion_classifier"):
        with torch.no_grad():
            bias_correction = model.fusion_classifier.bias.mean().item()
            model.fusion_classifier.bias -= bias_correction
        print("Bias correction applied after epoch.")
    return epoch_loss / batch_count, epoch_acc / batch_count

def evm_evaluate(model, loader, global_classes, evm_tailsize=0.3, evm_threshold=0.7):
    feature_dict, features, labels = extract_features_for_evm(model, loader, DEVICE, global_classes)
    evm = RegularizedEVMClassifier(tailsize=evm_tailsize, cover_threshold=evm_threshold,lambda_reg=0.1)
    evm.fit(feature_dict)
    preds, _ = evm.predict(features, threshold=evm_threshold)

    # Convert numeric labels to string class names if applicable
    if isinstance(global_classes[0], str):
        label_strs = [global_classes[l] for l in labels]
    else:
        label_strs = labels

    known_mask = [p != 'unknown' for p in preds]
    correct = [1 if p == gt else 0 for p, gt in zip(preds, label_strs)]
    accuracy = np.sum(np.array(correct)[known_mask]) / max(np.sum(known_mask), 1)
    print(f"EVM Accuracy (known only): {accuracy:.4f}  (threshold={evm_threshold})")
    return accuracy, preds


def run_domain_incremental_with_evm_training(
    data_root,
    ocr_tensor_dirs,
    domains,
    global_classes,
    eaml_ckpt_path,
    checkpoint_dir,
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
    evm_tailsize=0.3,
    evm_threshold=0.7,
    lambda_evm=0.0
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    class_to_idx = {cls: idx for idx, cls in enumerate(global_classes)}
    inc_strategy = DistillationDomainIL(DEVICE, temperature, lambda_distill) if strategy == "distillation" else StandardDomainIL(DEVICE)
    exemplar_mgr = ExemplarManager(max_exemplars=max_exemplars, selection_strategy=exemplar_selection) if use_exemplars else None
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
    train_loader = dil_loader.get_domain_loaders('train').get(incremental_domain)
    val_loader = dil_loader.get_domain_loaders('val').get(incremental_domain)
    test_loaders = dil_loader.get_domain_loaders('test')
    test_loader_pretrained = test_loaders.get(pretrained_domain)
    test_loader_incremental = test_loaders.get(incremental_domain)
    old_classes = 16
    new_classes = len(global_classes)
    model = load_eaml_model_partial(
        eaml_ckpt_path,
        old_classes,
        new_classes,
        device=DEVICE,
        text_branch=True
    )
    model = set_finetune_mode(model, finetune_mode, unfreeze_depth)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    lr_scheduler = AdaptiveLR(optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    ewc = EWC(model, test_loader_pretrained, DEVICE, lambda_ewc) if use_ewc else None
    old_model = load_eaml_model_partial(
        eaml_ckpt_path,
        old_classes,
        new_classes,
        device=DEVICE,
        text_branch=True
    )
    old_model.eval()
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_model_path = None
    no_improve = 0
    start_epoch = 1

    # EVM initialization for hybrid loss
    evm_hybrid = RegularizedEVMClassifier(tailsize=evm_tailsize, cover_threshold=evm_threshold,lambda_reg=0.1)
    # Initial EVM fit on seed train
    feature_dict, _, _ = extract_features_for_evm(model, train_loader, DEVICE, global_classes)
    evm_hybrid.fit(feature_dict)
    if resume and resume_ckpt_path is not None and os.path.exists(resume_ckpt_path):
        print(f"Resuming training from checkpoint: {resume_ckpt_path}")
        checkpoint = torch.load(resume_ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        no_improve = checkpoint.get('no_improve', 0)
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
        print(f"Resumed at epoch {start_epoch}, patience={no_improve}, best_val_acc={best_val_acc}, best_val_loss={best_val_loss}")
    print("Dataset overview:")
    counts = dil_loader.get_class_counts()
    for domain, count in counts.items():
        print(f"  Domain '{domain}': {count} classes")
    print("=== Training Starting ===")
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        train_loader_dict = {incremental_domain: train_loader}
        train_loss, train_acc = train_one_epoch_dil_evm(
            model=model,
            train_loaders=train_loader_dict,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            strategy=inc_strategy,
            global_classes=global_classes,
            exemplar_manager=exemplar_mgr,
            ewc=ewc,
            old_model=old_model if isinstance(inc_strategy, DistillationDomainIL) else None,
            use_bias_correction=use_bias_correction,
            evm=evm_hybrid,
            lambda_evm=lambda_evm
        )
        val_loss, val_acc = evaluate_dil(model, {incremental_domain: val_loader}, DEVICE)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        is_best = (val_acc > best_val_acc) or (val_loss < best_val_loss)
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            print(f"Patience counter: {no_improve}/{patience}")
            if no_improve >= patience:
                print("Early stopping triggered")
                break
        extra_data = {
            "no_improve": no_improve,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
        }
        if is_best:
            save_checkpoint_dil(model, optimizer, epoch, os.path.join(checkpoint_dir, "best_model.pth"), extra_data=extra_data)
        save_epoch_checkpoint_dil(model, optimizer, epoch, checkpoint_dir, is_best=is_best, extra_data=extra_data, max_keep_last=2)

        if is_best:
            print(f"Validation improved at epoch {epoch}, evaluating test sets...")
            pretrained_acc, _ = evaluate_domain(model, test_loader_pretrained, DEVICE, len(global_classes), global_classes)
            print(f"Test accuracy on pretrained domain '{pretrained_domain}': {pretrained_acc:.4f}")

            incremental_acc, _ = evaluate_domain(model, test_loader_incremental, DEVICE, len(global_classes), global_classes)
            print(f"Test accuracy on incremental domain '{incremental_domain}': {incremental_acc:.4f}")
        if is_best:
            print(f"Validation improved at epoch {epoch}, evaluating test sets with EVM...")
            for domain_name, loader in [(pretrained_domain, test_loader_pretrained), (incremental_domain, test_loader_incremental)]:
                accuracy, _ = evm_evaluate(model, loader, global_classes, evm_tailsize, evm_threshold)
                print(f"RegEVM Test accuracy on domain '{domain_name}': {accuracy:.4f}")
        lr_scheduler.step({'accuracy': val_acc, 'loss': val_loss})
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    
    print("Final evaluation on test datasets:")

    for domain_name, loader in [(pretrained_domain, test_loader_pretrained), (incremental_domain, test_loader_incremental)]:
        acc, _ = evaluate_domain(model, loader, DEVICE, len(global_classes), global_classes)
        print(f"Test accuracy on domain '{domain_name}': {acc:.4f}")

    print("Final RegEVM evaluation on test datasets:")
    for domain_name, loader in [(pretrained_domain, test_loader_pretrained), (incremental_domain, test_loader_incremental)]:
        accuracy, _ = evm_evaluate(model, loader, global_classes, evm_tailsize, evm_threshold)
        print(f"Final RegEVM accuracy on domain '{domain_name}': {accuracy:.4f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--ocr_tensor_dirs', nargs=2, required=True, help='Two tensor dirs matching domains')
    p.add_argument('--domains', required=True, help='Comma-separated domain names')
    p.add_argument('--global_classes', required=True, help='Comma-separated global class names')
    p.add_argument('--eaml_ckpt_path', required=True)
    p.add_argument('--checkpoint_dir', required=True)
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
    p.add_argument('--resume', action='store_true', help="Resume training from last checkpoint.")
    p.add_argument('--resume_ckpt_path', type=str)
    p.add_argument('--evm_tailsize', type=float, default=0.5)
    p.add_argument('--evm_threshold', type=float, default=0.7)
    p.add_argument('--lambda_evm', type=float, default=0.1)
    args = p.parse_args()
    run_domain_incremental_with_evm_training(
        data_root=args.data_dir,
        ocr_tensor_dirs={d: t for d, t in zip(args.domains.split(','), args.ocr_tensor_dirs)},
        domains=args.domains.split(','),
        global_classes=args.global_classes.split(','),
        eaml_ckpt_path=args.eaml_ckpt_path,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=0.0,
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
        evm_tailsize=args.evm_tailsize,
        evm_threshold=args.evm_threshold,
        lambda_evm=args.lambda_evm
    )
