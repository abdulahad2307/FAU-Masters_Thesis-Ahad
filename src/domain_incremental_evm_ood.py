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
    StandardDomainIL, DistillationDomainIL, EWC, ExemplarManager, AdaptiveLR, extract_features_by_class
)
from utils.domain_IL.dil_model_loader import load_eaml_model_partial, set_finetune_mode

from utils.evm.evm_classifier import EVMClassifier
from utils.evm.evm_eval import evm_openset_metrics
from utils.evm.evm_viz import plot_openset_histograms
from utils.ood.msp import MSP_OOD
from utils.ood.vim import VIM_OOD
from utils.ood.gradnorm import GradNorm_OOD
from utils.ood.ood_viz import plot_ood_histograms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_domain_classwise(model, loader, global_classes, domain_name, device):
    acc = evaluate_dil(model, {domain_name: loader}, device)
    class_acc = classwise_accuracy(model, loader, device, len(global_classes))
    print(f"\nDomain: {domain_name} - Overall Acc: {acc:.4f}")
    for i, cacc in enumerate(class_acc):
        print(f"  Class {global_classes[i]}: {cacc:.4f}")
    return acc, class_acc

def train_one_epoch_dil_with_evm_ood(
    model,
    train_loaders,
    optimizer,
    criterion,
    device,
    strategy,
    exemplar_manager=None,
    ewc=None,
    old_model=None,
    use_bias_correction=True,
    evm=None,
    ood_detector=None,
    lambda_evm=0.1,
    lambda_ood=0.1,
):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for domain_name, train_loader in train_loaders.items():
        for batch in train_loader:
            optimizer.zero_grad()

            images = batch["images"].to(device)
            texts = batch.get("texts")
            input_ids = batch["texts"]["input_ids"].to(device)
            attention_mask = batch["texts"]["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(images=images, input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            features = model.extract_features(images=images, input_ids=input_ids, attention_mask=attention_mask, texts=texts)

            inputs_for_ood = {
                "images": images,
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

            loss = criterion(logits, labels)

            # EVM loss
            if evm is not None and hasattr(evm, 'initialized') and evm.initialized and features is not None:
                evm_probs = evm.predict_proba_tensor(features)
                current_classes = list(evm.weibull_models.keys())
                label_to_local_idx = {cls: idx for idx, cls in enumerate(current_classes)}
                local_labels = torch.tensor([label_to_local_idx[int(l)] for l in labels.cpu()])

                batch_indices = torch.arange(labels.size(0)).cpu()
                local_labels = local_labels.cpu()

                true_class_probs = evm_probs[batch_indices, local_labels]

                evm_loss = -torch.log(true_class_probs + 1e-8).mean()
                loss = loss + lambda_evm * evm_loss


            # OOD loss
            if ood_detector is not None and hasattr(ood_detector, 'initialized') and ood_detector.initialized and inputs_for_ood is not None:
                with torch.no_grad():
                    if hasattr(ood_detector, "score_batch"):
                        ood_scores = ood_detector.score_batch(model, inputs_for_ood, device, texts=texts)
                        if ood_scores is not None:
                            ood_scores_tensor = torch.tensor(ood_scores, device=device)
                            ood_loss = ood_scores_tensor.mean()
                            loss = loss + lambda_ood * ood_loss

            # Incremental strategy loss
            if strategy:
                inc_loss, preds, target_labels = strategy.compute_loss(
                    model, batch, criterion, old_model=old_model, ewc=ewc
                )
                loss = loss + inc_loss
            else:
                preds = torch.argmax(logits, dim=1)
                target_labels = labels

            loss.backward()
            optimizer.step()

            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(target_labels.detach().cpu().tolist())
            total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, acc

def run_domain_incremental_evm_ood(
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
    lambda_evm=0.1,
    lambda_ood=0.1,
    ood_method="msp",
    ood_threshold=0.75,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    class_to_idx = {cls: idx for idx, cls in enumerate(global_classes)}

    inc_strategy = (
        DistillationDomainIL(DEVICE, temperature, lambda_distill)
        if strategy == "distillation" else StandardDomainIL(DEVICE)
    )

    exemplar_mgr = (
        ExemplarManager(max_exemplars=max_exemplars, selection_strategy=exemplar_selection)
        if use_exemplars else None
    )

    OOD_CLASSES = {"msp": MSP_OOD, "vim": VIM_OOD, "gradnorm": GradNorm_OOD}
    ood_cls = OOD_CLASSES.get(ood_method.lower())
    if ood_cls is None:
        raise ValueError(f"OOD method {ood_method} not implemented")
    ood_detector = ood_cls(threshold=ood_threshold)
    evm = EVMClassifier(tailsize=lambda_evm, cover_threshold=0.7)

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
        eaml_ckpt_path, old_classes, new_classes, device=DEVICE, text_branch=True
    )
    model = set_finetune_mode(model, finetune_mode, unfreeze_depth)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )
    lr_scheduler = AdaptiveLR(optimizer)
    criterion = nn.CrossEntropyLoss()

    ewc = EWC(model, test_loader_pretrained, DEVICE, lambda_ewc) if use_ewc else None

    old_model = load_eaml_model_partial(
        eaml_ckpt_path, old_classes, new_classes, device=DEVICE, text_branch=True
    )
    old_model.eval()

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    no_improve = 0
    start_epoch = 1

    if resume and resume_ckpt_path is not None and os.path.exists(resume_ckpt_path):
        print(f"Resuming from: {resume_ckpt_path}")
        checkpoint = torch.load(resume_ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        no_improve = checkpoint.get('no_improve', 0)
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        train_loader_dict = {incremental_domain: train_loader}
        train_loss, train_acc = train_one_epoch_dil_with_evm_ood(
            model=model,
            train_loaders=train_loader_dict,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            strategy=inc_strategy,
            exemplar_manager=exemplar_mgr,
            ewc=ewc,
            old_model=old_model if isinstance(inc_strategy, DistillationDomainIL) else None,
            use_bias_correction=use_bias_correction,
            evm=evm,
            ood_detector=ood_detector,
            lambda_evm=lambda_evm,
            lambda_ood=lambda_ood,
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
            save_checkpoint_dil(model, optimizer, epoch, best_model_path, extra_data=extra_data)
        save_epoch_checkpoint_dil(model, optimizer, epoch, checkpoint_dir, is_best=is_best, extra_data=extra_data, max_keep_last=2)

        if is_best:
            print(f"Validation improved at epoch {epoch}, evaluating test sets and OOD/EVM metrics...")
            # Fit EVM on training features
            train_feats = extract_features_by_class(model, train_loader, DEVICE)
            """
            all_features = []
            model.eval()
            with torch.no_grad():
                for batch in train_loader:
                    images = batch["images"].to(DEVICE)
                    input_ids = batch["texts"]["input_ids"].to(DEVICE)
                    attention_mask = batch["texts"]["attention_mask"].to(DEVICE)
                    feats = model.extract_features_by_class(
                        images=images, input_ids=input_ids, attention_mask=attention_mask, texts=batch["texts"]
                    )
                    all_features.append(feats.cpu())
            train_feats = torch.cat(all_features, dim=0)
            """
            evm = evm.fit(train_feats)
            # Fit OOD detector on val features
            val_feats = extract_features_by_class(model, val_loader, DEVICE)
            """
            all_features = []
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["images"].to(DEVICE)
                    input_ids = batch["texts"]["input_ids"].to(DEVICE)
                    attention_mask = batch["texts"]["attention_mask"].to(DEVICE)
                    feats = model.extract_features_by_class(
                        images=images, input_ids=input_ids, attention_mask=attention_mask, texts=batch["texts"]
                    )
                    all_features.append(feats.cpu())
            val_feats = torch.cat(all_features, dim=0)
            """
            ood_detector.fit(val_feats)

            evm_res = evm_openset_metrics(evm,val_feats, global_classes)
            print(f"[EVM val] Acc: {evm_res['open_set_accuracy']:.4f}, Reject: {evm_res['unknown_rejection']:.4f}")

            ood_res = ood_detector.ood_metrics(val_feats, global_classes)
            print(f"[OOD {ood_detector.__class__.__name__} val] Acc: {ood_res['open_set_accuracy']:.4f}, Rejection: {ood_res['unknown_rejection']:.4f}")

            #plot_ood_histograms(ood_res["y_true"], ood_res["y_pred"], ood_res["scores"], savepath=os.path.join(checkpoint_dir, f"ood_hist_epoch{epoch}.png"))
            #plot_openset_histograms(evm,val_feats,evm_res["scores"],savepath=os.path.join(checkpoint_dir, f"evm_hist_epoch{epoch}.png"))

            for domain_name, loader in [(pretrained_domain, test_loader_pretrained), (incremental_domain, test_loader_incremental)]:
                acc, _ = evaluate_domain(model, loader, DEVICE, len(global_classes), global_classes)
                print(f"Test accuracy on domain '{domain_name}': {acc:.4f}")

        lr_scheduler.step({'accuracy': val_acc, 'loss': val_loss})

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    print("Final evaluation on test datasets:")

    for domain_name, loader in [(pretrained_domain, test_loader_pretrained), (incremental_domain, test_loader_incremental)]:
        acc, _ = evaluate_domain(model, loader, DEVICE, len(global_classes), global_classes)
        print(f"Test accuracy on domain '{domain_name}': {acc:.4f}")

    test_feats = extract_features_by_class(model, test_loader_incremental, DEVICE)
    evm_test_res = evm.evm_openset_metrics(test_feats, global_classes)
    print(f"[EVM test] Acc: {evm_test_res['open_set_accuracy']:.4f}, Reject: {evm_test_res['unknown_rejection']:.4f}")

    ood_test_res = ood_detector.ood_metrics(test_feats, global_classes)
    print(f"[OOD {ood_detector.__class__.__name__} test] Acc: {ood_test_res['open_set_accuracy']:.4f}, Rejection: {ood_test_res['unknown_rejection']:.4f}")

    plot_ood_histograms(ood_test_res["y_true"], ood_test_res["y_pred"], ood_test_res["scores"], savepath=os.path.join(checkpoint_dir, "ood_hist_final.png"))
    plot_openset_histograms(evm, test_feats, savepath=os.path.join(checkpoint_dir, "evm_hist_final.png"))

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
    p.add_argument('--lambda_evm', type=float, default=0.1)
    p.add_argument('--lambda_ood', type=float, default=0.1)
    p.add_argument('--ood_method', type=str, default='msp', choices=['msp', 'vim', 'gradnorm'])
    p.add_argument('--ood_threshold', type=float, default=0.75)

    args = p.parse_args()
    run_domain_incremental_evm_ood(
        data_root=args.data_dir,
        ocr_tensor_dirs={d: t for d, t in zip(args.domains.split(','), args.ocr_tensor_dirs)},
        domains=args.domains.split(','),
        #global_classes=args.global_classes.split(','),
        global_classes = [c.strip().replace('{','').replace('}','') for c in args.global_classes.split(',')],
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
        lambda_evm=args.lambda_evm,
        lambda_ood=args.lambda_ood,
        ood_method=args.ood_method,
        ood_threshold=args.ood_threshold
    )
