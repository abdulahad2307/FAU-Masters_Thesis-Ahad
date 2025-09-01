import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Dict

from sklearn.metrics import precision_score, recall_score, f1_score

from utils.eaml.eaml_model import EAMLModel
from utils.docformer.model import DocFormer
from utils.docformer.config import DocFormerConfig
from utils.class_IL.dataloader_utils import (
    get_class_il_loader, EAMLClassILDataset, common_transform, eaml_collate_fn
)
from utils.class_IL.train_utils import (
    save_checkpoint, load_checkpoint, CILMetrics, evaluate
)
from utils.class_IL.cil_utils import (
    StandardIncremental, DistillationIncremental, EWC, ExemplarManager,
    AdaptiveLR, extract_features, extract_features_and_logits, extract_feature_vectors
)
from utils.class_IL.training_modes import get_training_mode
from utils.evm.evm_classifier import EVMClassifier
from utils.evm.evm_eval import evm_openset_metrics
from utils.evm.evm_viz import plot_openset_histograms
from utils.ood.msp import MSP_OOD
from utils.ood.vim import VIM_OOD
from utils.ood.gradnorm import GradNorm_OOD
from utils.ood.ood_viz import plot_ood_histograms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    else:
        return data


def train_one_epoch_evm_ood(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    metrics,
    inc_strategy,
    evm=None,
    ood_detector=None,
    lambda_evm=0.1,
    lambda_ood=0.1,
    old_model=None,
    ewc=None
):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        # Supporting original batch structure with dictionary keys or tuple/list
        if isinstance(batch, dict) and ("images" in batch or "pixel_values" in batch):
            if "images" in batch:
                images = batch["images"].to(device)
                texts = batch["texts"]
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
            else:
                inputs = {
                    "pixel_values": batch["pixel_values"].to(device),
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "bboxes": batch["bboxes"].to(device)
                }
                labels = batch["labels"].to(device)
                texts = batch["texts"]
                outputs = model(**inputs, task="classification")
                logits = outputs["logits"]
                features = model.extract_features(**inputs)
                inputs_for_ood = inputs
        else:
            # fallback for non-dict batches or unknown keys
            inputs_for_ood = None
            labels = batch[1].to(device)
            if hasattr(model, 'extract_features'):
                features = model.extract_features(batch[0].to(device))
            else:
                features = None
            outputs = model(batch[0].to(device))
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs

        loss = criterion(logits, labels)

        # EVM loss
        if evm is not None and evm.initialized and features is not None:
            evm_probs = evm.predict_proba_tensor(features)  # returns cpu tensor
            batch_indices = torch.arange(labels.size(0))
            true_class_probs = evm_probs[batch_indices, labels.cpu()]
            evm_loss = -torch.log(true_class_probs + 1e-8).mean()
            loss = loss + lambda_evm * evm_loss

        # OOD loss - batch-wise scoring if supported
        if ood_detector is not None and ood_detector.initialized and inputs_for_ood is not None:
            with torch.no_grad():
                if hasattr(ood_detector, "score_batch"):
                    ood_scores = ood_detector.score_batch(model, inputs_for_ood, device, texts=texts)
                    if ood_scores is not None:
                        ood_scores_tensor = torch.tensor(ood_scores, device=device)
                        ood_loss = ood_scores_tensor.mean()
                        loss = loss + lambda_ood * ood_loss

        # Incremental strategy loss and predictions
        if inc_strategy:
            inc_loss, preds, target_labels = inc_strategy.compute_loss(
                model, batch, criterion, old_model=old_model, ewc=ewc
            )
            loss = loss + inc_loss if evm is None else loss + inc_loss
            preds = preds
            target_labels = target_labels
        else:
            preds = torch.argmax(logits, dim=1)
            target_labels = labels

        loss.backward()
        optimizer.step()

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(target_labels.detach().cpu().tolist())
        metrics.update(np.array(all_preds), np.array(all_labels))
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    acc = metrics.get_accuracy() if hasattr(metrics, "get_accuracy") else metrics.get_metrics().get("top1_acc", 0)

    print(f"Train Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")

    return {"loss": avg_loss, "accuracy": acc}


def run_incremental_learning_evm_ood(
    data_root: str,
    ocr_tensor_path: str,
    all_classes: List[str],
    base_classes: List[str],
    unseen_classes: List[str],
    base_model_path: str,
    model_name: str,
    checkpoint_dir: str,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    num_epochs: int = 10,
    strategy: str = "distillation",
    temperature: float = 2.0,
    lambda_distill: float = 1.0,
    lambda_ewc: float = 5000.0,
    use_ewc: bool = True,
    use_exemplars: bool = True,
    max_exemplars: int = 320,
    exemplar_selection: str = "herding",
    training_mode: str = "last_layer",
    trainable_layers: Optional[List[str]] = None,
    resume: bool = False,
    resume_checkpoint: Optional[str] = None,
    global_best_acc: float = 0.0,
    full_model_acc: Optional[float] = None,
    patience: int = 10,
    use_balanced_sampler: bool = True,
    use_bias_correction: bool = True,
    lambda_evm: float = 0.1,
    lambda_ood: float = 0.1,
    ood_method: str = "msp",
    ood_threshold: float = 0.75,
):
    import shutil
    from torch.utils.data import WeightedRandomSampler, DataLoader


    os.makedirs(checkpoint_dir, exist_ok=True)


    if strategy == "distillation":
        inc_strategy = DistillationIncremental(DEVICE, temperature, lambda_distill)
    else:
        inc_strategy = StandardIncremental(DEVICE)


    exemplar_mgr = (
        ExemplarManager(max_exemplars=max_exemplars, selection_strategy=exemplar_selection)
        if use_exemplars
        else None
    )
    if unseen_classes is None or len(unseen_classes) == 0:
        raise ValueError("Must provide a non-empty list of unseen_classes")
    for cls in unseen_classes:
        if cls not in all_classes:
            raise ValueError(f"Unseen class {cls} not in all_classes")
        if cls in base_classes:
            raise ValueError(f"Unseen class {cls} already in base_classes")


    current_classes = base_classes.copy()
    ewc = None
    evm = EVMClassifier(tailsize=lambda_evm, cover_threshold=0.7)  # keep the original threshold
    OOD_CLASSES = {"msp": MSP_OOD, "vim": VIM_OOD, "gradnorm": GradNorm_OOD}
    ood_cls = OOD_CLASSES.get(ood_method.lower())
    if ood_cls is None:
        raise ValueError(f"OOD method {ood_method} not implemented")
    ood_detector = ood_cls(threshold=ood_threshold)


    start_idx = 0
    start_epoch = 0
    epochs_no_improve = 0
    best_acc = 0.0
    best_path = None
    final_best_path = None


    if resume and resume_checkpoint and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=DEVICE,weights_only=False)
        current_classes = checkpoint.get("current_classes", base_classes.copy())
        start_idx = checkpoint.get("unseen_index", 0)
        start_epoch = checkpoint.get("epoch", 0)
        epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
        best_acc = checkpoint.get("step_best_acc", 0.0)
        best_path = checkpoint.get("step_best_path", None)
        print(f"Resuming from index {start_idx}, epoch {start_epoch}")
        del checkpoint
        torch.cuda.empty_cache()


    for unseen_idx, new_class in enumerate(unseen_classes[start_idx:], start_idx):
        if unseen_idx > start_idx:
            start_epoch = 0
            epochs_no_improve = 0
            best_acc = 0.0
            best_path = None


        previous_classes = current_classes.copy()
        if new_class not in current_classes:
            current_classes.append(new_class)
        print(f"\n= Incremental step {unseen_idx+1}/{len(unseen_classes)} - Adding {new_class} =")


        replay_samples = []
        if use_exemplars and exemplar_mgr is not None and unseen_idx > 0:
            replay_samples = exemplar_mgr.get_exemplar_dataset()
        new_data_samples = []
        train_dataset = None
        train_loader = None

        if model_name == "eaml":
            train_dataset = EAMLClassILDataset(
                data_dir=os.path.join(data_root, "train"),
                current_classes=current_classes,
                ocr_data_path=ocr_tensor_path,
                transform=common_transform
            )
            if use_exemplars and replay_samples:
                new_data_samples = [s for s in train_dataset.samples if s[2] == new_class]
                for i, s in enumerate(replay_samples + new_data_samples):
                    assert isinstance(s, tuple) and len(s) == 3 and isinstance(s[2], str), \
                        f"Sample at position {i} is not correct tuple: {type(s)}, {getattr(s, 'keys', lambda: None)() if isinstance(s, dict) else s}"
                train_dataset.samples = replay_samples + new_data_samples
                print(f"Using {len(replay_samples)} exemplars and {len(new_data_samples)} new samples for training.")
            if use_balanced_sampler:
                from collections import Counter
                class_counts = Counter([s[2] for s in train_dataset.samples])
                total = sum(class_counts.values())
                class_weights = {cls: total/count for cls, count in class_counts.items()}
                sample_weights = [class_weights[s[2]] for s in train_dataset.samples]
                sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    sampler=sampler,
                    num_workers=4,
                    collate_fn=eaml_collate_fn
                )
            else:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4,
                    collate_fn=eaml_collate_fn
                )
        else:
            train_loader = get_class_il_loader(model_name, os.path.join(data_root, "train"), current_classes,
                                               batch_size, ocr_data=ocr_tensor_path)
        val_loader = get_class_il_loader(model_name, os.path.join(data_root, "val"), current_classes,
                                         batch_size, ocr_data=ocr_tensor_path)
        test_loader = get_class_il_loader(model_name, os.path.join(data_root, "test"), current_classes,
                                          batch_size, ocr_data=ocr_tensor_path)
        torch.cuda.empty_cache()


        if model_name == "docformer":
            cfg = DocFormerConfig()
            model = DocFormer(cfg).to(DEVICE)
        else:
            model = EAMLModel(num_classes=len(current_classes)).to(DEVICE)


        # Modified weight loading:
        if resume is False or (resume and unseen_idx != start_idx):
            if os.path.exists(base_model_path):
                checkpoint = torch.load(base_model_path, map_location=DEVICE,weights_only=False)
                loaded_state = checkpoint.get("model_state_dict", checkpoint)
                model_state = model.state_dict()
                compatible_state = {}
                for k, v in loaded_state.items():
                    if k in model_state:
                        if v.shape == model_state[k].shape:
                            compatible_state[k] = v
                        else:
                            print(f"Skipping {k} due to size mismatch: {v.shape} vs {model_state[k].shape}")
                    else:
                        print(f"Skipping {k} as not found in current model")
                model_state.update(compatible_state)
                model.load_state_dict(model_state)
                del checkpoint
                torch.cuda.empty_cache()


        start_ep = start_epoch if unseen_idx == start_idx else 0
        ep_no_improve = epochs_no_improve if unseen_idx == start_idx else 0
        best_acc_local = best_acc if unseen_idx == start_idx else 0.0
        best_path_local = best_path if unseen_idx == start_idx else None


        if resume and unseen_idx == start_idx and resume_checkpoint and os.path.exists(resume_checkpoint):
            checkpoint = load_checkpoint(model, None, resume_checkpoint, DEVICE)
            del checkpoint
            torch.cuda.empty_cache()

        """
        old_model = None
        if len(previous_classes) > 0:
            if model_name == "docformer":
                old_model = DocFormer(cfg).to(DEVICE)
            else:
                old_model = EAMLModel(num_classes=len(previous_classes)).to(DEVICE)
            ckpt_path = best_path_local if best_path_local is not None else base_model_path
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=DEVICE,weights_only=False)
                old_model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
                del ckpt
                torch.cuda.empty_cache()
            old_model.eval()
            model = inc_strategy.adapt_model(model, len(previous_classes), len(current_classes), model_name).to(DEVICE)
        """
        
        old_model = None

        if len(previous_classes) > 0:
            ckpt_path = resume_checkpoint if resume and unseen_idx == start_idx else (
                best_path_local if best_path_local else base_model_path)
            ckpt = torch.load(ckpt_path, map_location=DEVICE,weights_only=False)
            if 'current_classes' in ckpt:
                prev_classes_in_ckpt = ckpt['current_classes']
            else:
                prev_classes_in_ckpt = previous_classes
            n_old_classes = len(prev_classes_in_ckpt)
            print(f"Constructing old_model with {n_old_classes} classes from {ckpt_path}")
            old_model = EAMLModel(num_classes=n_old_classes).to(DEVICE)
            old_model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
            del ckpt
            torch.cuda.empty_cache()
            old_model.eval()
            model = inc_strategy.adapt_model(model, n_old_classes, len(current_classes), model_name).to(DEVICE)




        tm = get_training_mode(model, training_mode, trainable_layers)
        model = tm.prepare_for_training()


        optimizer = torch.optim.AdamW(
            tm.get_trainable_params(), lr=lr, weight_decay=weight_decay
        )


        if resume and unseen_idx == start_idx and resume_checkpoint and os.path.exists(resume_checkpoint):
            ckpt = torch.load(resume_checkpoint, map_location=DEVICE,weights_only=False)
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            del ckpt
            torch.cuda.empty_cache()


        lr_scheduler = AdaptiveLR(optimizer, base_lr=lr)
        criterion = nn.CrossEntropyLoss()


        patience_counter = ep_no_improve


        for epoch in range(start_ep, num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs} for class {new_class}")


            metrics = CILMetrics(current_classes)
            train_results = train_one_epoch_evm_ood(
                model, train_loader, optimizer, criterion, DEVICE, metrics, inc_strategy, evm, ood_detector, lambda_evm, lambda_ood, old_model, ewc if use_ewc else None
            )


            val_metrics = CILMetrics(current_classes)
            val_results = evaluate(model, val_loader, DEVICE, val_metrics, None)


            print(f"Train loss: {train_results['loss']:.4f} | Train acc: {train_results['accuracy']:.4f}")
            print(f"Val loss: {val_results['loss']:.4f} | Val acc: {val_results['top1_acc']:.4f}")


            path_epoch = os.path.join(checkpoint_dir, f"epoch{epoch + 1}_{new_class}.pth")
            path_best = os.path.join(checkpoint_dir, f"best_model_{new_class}.pth")


            if val_results["top1_acc"] > best_acc_local:
                best_acc_local = val_results["top1_acc"]
                patience_counter = 0
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    path_best,
                    extra_data={
                        "current_classes": current_classes,
                        "unseen_index": unseen_idx,
                        "epochs_no_improve": patience_counter,
                        "step_best_acc": best_acc_local,
                        "step_best_path": path_best,
                        "epoch": epoch + 1,
                        "unseen_class": new_class,
                    },
                )

                #ckpt_dir = os.path.dirname(path_best)
                for fname in os.listdir(checkpoint_dir):
                    fpath = os.path.join(checkpoint_dir, fname)
                    # Skipping deletion if this is the best checkpoint, or if it's not a .pth file
                    if fpath == path_best or not fname.endswith('.pth'):
                        continue
                    try:
                        os.remove(fpath)
                    except Exception as e:
                        print(f"Failed to remove {fpath}: {e}")
                
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    path_epoch,
                    extra_data={
                        "current_classes": current_classes,
                        "unseen_index": unseen_idx,
                        "epochs_no_improve": patience_counter,
                        "step_best_acc": best_acc_local,
                        "step_best_path": best_path,
                        "epoch": epoch + 1,
                        "unseen_class": new_class,
                    },
                )

                
                test_metrics = CILMetrics(current_classes)
                test_results = evaluate(model, test_loader, DEVICE, test_metrics, full_model_acc)
                print("Class-wise Test accuracy:")
                for c, a in zip(current_classes, test_results["class_acc"]):
                    print(f"  {c}: {a:.4f}")
                
                # ===== GIL =====
                gil_base = 0.953
                if full_model_acc is not None:
                    gil = (test_results['top1_acc'] - full_model_acc) / (1 - full_model_acc)
                    print(f"GIL: {gil:.4f}")
                if full_model_acc is not None:
                    gil = (test_results['top1_acc'] - gil_base) / (1 - gil_base)
                    print(f"GIL (wrt base): {gil:.4f}")

                val_feats = extract_features(model, val_loader, DEVICE)


                if ood_method.lower() == "vim":
                    fes = {}
                    lgs = {}
                    for cls in current_classes:
                        cls_loader = get_class_il_loader(model_name, os.path.join(data_root, "train"), [cls], batch_size, ocr_data=ocr_tensor_path)
                        fts, lgts = extract_features_and_logits(model, cls_loader, DEVICE)
                        fes[cls] = fts
                        lgs[cls] = lgts
                    ood_detector.fit(fes, lgs)
                else:
                    train_feats = extract_features(model, train_loader, DEVICE)
                    ood_detector.fit(train_feats)


                ood_res = ood_detector.ood_metrics(val_feats, current_classes)
                print(f"[OOD {ood_detector.__class__.__name__} val] Acc: {ood_res['open_set_accuracy']:.4f}, Rejection: {ood_res['unknown_rejection']:.4f}")


                if use_exemplars and evm.initialized:
                    evm_feats = extract_features(model, val_loader, DEVICE)
                    evm_res = evm.ood_metrics(evm_feats, current_classes)
                    print(f"[EVM val] Acc: {evm_res['open_set_accuracy']:.4f}, Reject: {evm_res['unknown_rejection']:.4f}")


            else:
                patience_counter += 1
                print(f"Patience counter: {patience_counter}/{patience}")
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    path_epoch,
                    extra_data={
                        "current_classes": current_classes,
                        "unseen_index": unseen_idx,
                        "epochs_no_improve": patience_counter,
                        "step_best_acc": best_acc_local,
                        "step_best_path": best_path,
                        "epoch": epoch + 1,
                        "unseen_class": new_class,
                    },
                )
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break


            lr_scheduler.step(val_results["top1_acc"])
            torch.cuda.empty_cache()


        if use_ewc:
            ewc = EWC(model, train_loader, DEVICE, lambda_ewc)


        if use_exemplars:
            exemplar_mgr.update(train_dataset, new_class, model)


        if use_bias_correction:
            if hasattr(model, "fusion_classifier"):
                with torch.no_grad():
                    bmean = model.fusion_classifier.bias.mean()
                    model.fusion_classifier.bias -= bmean


        train_feats = extract_features(model, train_loader, DEVICE)
        evm.fit(train_feats)


        if best_path is not None:
            base_model_path = best_path
        else:
            print("Warning: No best checkpoint found, skipping base_model_path update.")


        if unseen_idx == len(unseen_classes) - 1:
            if best_path is not None:
                final_best_path = os.path.join(checkpoint_dir, "final_best_model.pth")
                shutil.copy(best_path, final_best_path)
                print(f"Saved final best model at: {final_best_path}")


    if final_best_path is not None:
        if model_name == "docformer":
            cfg = DocFormerConfig()
            final_model = DocFormer(cfg).to(DEVICE)
        else:
            final_model = EAMLModel(num_classes=len(all_classes)).to(DEVICE)


        checkpoint = torch.load(final_best_path, map_location=DEVICE,weights_only=False)
        final_model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        final_model.eval()
        torch.cuda.empty_cache()


        train_loader_full = get_class_il_loader(model_name, os.path.join(data_root, "train"), all_classes, batch_size, ocr_data=ocr_tensor_path)
        val_loader_full = get_class_il_loader(model_name, os.path.join(data_root, "val"), all_classes, batch_size, ocr_data=ocr_tensor_path)
        test_loader_full = get_class_il_loader(model_name, os.path.join(data_root, "test"), all_classes, batch_size, ocr_data=ocr_tensor_path)


        def evaluate_print(split_name, loader):
            metric = CILMetrics(all_classes)
            res = evaluate(final_model, loader, DEVICE, metric, full_model_acc)
            p = precision_score(res['labels'], res['preds'], average='macro', zero_division=0)
            r = recall_score(res['labels'], res['preds'], average='macro', zero_division=0)
            f = f1_score(res['labels'], res['preds'], average='macro', zero_division=0)
            print(f"{split_name} - Loss: {res['loss']:.4f} | Acc: {res['top1_acc']:.4f} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f:.4f}")
            if full_model_acc:
                gil = (res['top1_acc'] - full_model_acc) / (1 - full_model_acc)
                print(f"{split_name} - G_IL: {gil:.4f}")
            if split_name == "Test" and "class_acc" in res:
                print("Class-wise test accuracy:")
                for cls, acc in zip(all_classes, res["class_acc"]):
                    print(f"  {cls}: {acc:.4f}")
                
            torch.cuda.empty_cache()


        evaluate_print("Train", train_loader_full)
        evaluate_print("Validation", val_loader_full)
        evaluate_print("Test", test_loader_full)


        test_feats = extract_features(final_model, test_loader_full, DEVICE)
        ood_res = ood_detector.ood_metrics(test_feats, all_classes)
        print(f"[OOD {ood_detector.__class__.__name__} test] Acc: {ood_res['open_set_accuracy']:.4f}, Reject: {ood_res['unknown_rejection']:.4f}")


        if evm.initialized:
            evm_res = evm.ood_metrics(test_feats, all_classes)
            print(f"[EVM test] Acc: {evm_res['open_set_accuracy']:.4f}, Reject: {evm_res['unknown_rejection']:.4f}")


        plot_ood_histograms(ood_res["y_true"], ood_res["y_pred"], ood_res["scores"], savepath=os.path.join(checkpoint_dir, "ood_hist.png"))
        plot_openset_histograms(evm, test_feats, savepath=os.path.join(checkpoint_dir, "evm_hist.png"))



if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--ocr_tensor_path', required=True)
    p.add_argument('--all_classes', required=True)
    p.add_argument('--base_classes', required=True)
    p.add_argument('--unseen_classes', required=True, help="Comma-separated list of unseen classes")
    p.add_argument('--base_model_path', required=True)
    p.add_argument('--model_name', choices=['eaml', 'docformer'], required=True)
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--ood_method', type=str, default='msp', choices=['msp', 'vim', 'gradnorm'])
    p.add_argument('--ood_threshold', type=float, default=0.75)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_epochs', type=int, default=10)
    p.add_argument('--strategy', choices=['standard','distillation'], default='distillation')
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--lambda_distill', type=float, default=1.0)
    p.add_argument('--lambda_ewc', type=float, default=5000.0)
    p.add_argument('--use_ewc', action='store_true')
    p.add_argument('--use_exemplars', action='store_true')
    p.add_argument('--max_exemplars', type=int, default=320)
    p.add_argument('--exemplar_selection', choices=['random','herding'], default='herding')
    p.add_argument('--training_mode', choices=['full','last_layer','selective'], default='last_layer')
    p.add_argument('--trainable_layers', nargs='+', default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--resume_checkpoint', type=str, default=None)
    p.add_argument('--global_best_acc', type=float, default=0.0)
    p.add_argument('--full_model_acc', type=float, default=0.953)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--use_balanced_sampler', action='store_true')
    p.add_argument('--use_bias_correction', action='store_true')
    p.add_argument('--lambda_evm', type=float, default=0.1)
    p.add_argument('--lambda_ood', type=float, default=0.1)

    args = p.parse_args()

    run_incremental_learning_evm_ood(
        data_root=args.data_dir,
        ocr_tensor_path=args.ocr_tensor_path,
        all_classes=args.all_classes.split(','),
        base_classes=args.base_classes.split(','),
        unseen_classes=args.unseen_classes.split(','),
        base_model_path=args.base_model_path,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        ood_method=args.ood_method,
        ood_threshold=args.ood_threshold,
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
        training_mode=args.training_mode,
        trainable_layers=args.trainable_layers,
        resume=args.resume,
        resume_checkpoint=args.resume_checkpoint,
        global_best_acc=args.global_best_acc,
        full_model_acc=args.full_model_acc,
        weight_decay=args.weight_decay,
        patience=args.patience,
        use_balanced_sampler=args.use_balanced_sampler,
        use_bias_correction=args.use_bias_correction,
        lambda_evm=args.lambda_evm,
        lambda_ood=args.lambda_ood
    )
