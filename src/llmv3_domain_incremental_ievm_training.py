import argparse
import os
import torch
import torch.nn.functional as F
import gc
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from utils.llmv3.llmv3_model_loader import LayoutLMv3
from utils.llmv3.llmv3_incremental_dataloader import get_incremental_dataloader
from utils.llmv3.llmv3_incremental_utils import (
    EWC, distillation_loss, BiasCorrectionLayer,
    ExemplarHandler, expand_classifier, evaluate
)
import torch.optim as optim
import torch.cuda.amp as amp
from tqdm import tqdm
#from utils.evm.evm_classifier import EVMClassifier
from utils.ievm.ievm import IncrementalEVM


def parse_args():
    parser = argparse.ArgumentParser(description="LayoutLMv3 Domain-Incremental Training w/ EVM")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ocr_tensor_path_base', required=True)
    parser.add_argument('--ocr_tensor_path_inc', required=True)
    parser.add_argument('--all_classes', required=True)
    parser.add_argument('--dataset_base', default='rvl_cdip', choices=['rvl_cdip', 'tobacco3482'])
    parser.add_argument('--dataset_inc', default='tobacco3482', choices=['rvl_cdip', 'tobacco3482'])
    parser.add_argument('--base_model_path', required=True)
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--use_ewc', action='store_true')
    parser.add_argument('--lambda_ewc', type=float, default=5000.0)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_exemplars', type=int, default=16)
    parser.add_argument('--exemplar_selection', default='herding', choices=['random', 'herding'])
    parser.add_argument('--training_mode', default='last_layer', choices=['last_layer', 'full_model'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_checkpoint')
    parser.add_argument('--full_model_acc', type=float, default=None)
    parser.add_argument('--images_per_class', type=int, help="Max images per class for incremental domain")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    # EVM-specific parameters
    parser.add_argument('--evm_tailsize', type=float, default=0.5)
    parser.add_argument('--evm_threshold', type=float, default=0.7)
    parser.add_argument('--lambda_evm', type=float, default=0.1)
    return parser.parse_args()

# >>>>>>>>>>>>>>>>>>>> EVM routines <<<<<<<<<<<<<<<<<<<<<<<
def extract_features_for_evm(model, loader, device, global_classes):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for batch in loader:
            for k in ['input_ids', 'attention_mask', 'bbox', 'pixel_values']:
                batch[k] = batch[k].to(device)
            # Forward pass, collecting features from penultimate (just before classifier)
            feats = model.extract_features(
                batch['input_ids'], batch['bbox'], batch['attention_mask'], batch['pixel_values']
            )

            features.append(feats.cpu().numpy())
            labels.append(batch['labels'].cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    feature_dict = {}
    for i, l in enumerate(labels):
        class_name = global_classes[l] if isinstance(global_classes[0], str) else l
        feature_dict.setdefault(class_name, []).append(features[i])
    for k in feature_dict:
        feature_dict[k] = np.stack(feature_dict[k], axis=0)
    return feature_dict, features, labels
"""
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
"""

def hybrid_loss(logits, features, labels, evm, criterion, global_classes, lambda_evm):
    ce_loss = criterion(logits, labels)
    features_np = features.detach().cpu().numpy()
    label_strs = [global_classes[l] if isinstance(global_classes[0], str) else l for l in labels.cpu().numpy()]
    evm_probs = evm.predict_proba(features_np)
    sorted_keys = sorted(evm_probs.keys())

    valid_indices = []
    valid_probs = []
    for i, lstr in enumerate(label_strs):
        if lstr in evm_probs:
            class_idx = sorted_keys.index(lstr)
            prob = evm_probs[lstr][i]
            valid_indices.append(class_idx)
            valid_probs.append(prob)
        else:
            pass
    
    if len(valid_probs) == 0:
        return ce_loss

    prob_matrix = np.stack([evm_probs[c] for c in sorted_keys], axis=1)
    prob_true = np.array(valid_probs)
    evm_loss = -np.log(prob_true + 1e-6).mean()
    evm_loss_tensor = torch.tensor(evm_loss, dtype=torch.float32, device=logits.device)
    return ce_loss + lambda_evm * evm_loss_tensor

def evm_evaluate(model, loader, global_classes, evm_tailsize=0.3, evm_threshold=0.7):
    feature_dict, features, labels = extract_features_for_evm(model, loader, model.device, global_classes)
    #evm = EVMClassifier(tailsize=evm_tailsize, cover_threshold=evm_threshold)
    evm = IncrementalEVM(tailsize=0.5, ev_budget=10, cover_threshold=0.7)
    evm.fit(feature_dict)
    preds, _ = evm.predict(features, threshold=evm_threshold)
    if isinstance(global_classes[0], str):
        label_strs = [global_classes[l] for l in labels]
    else:
        label_strs = labels
    known_mask = [p != 'unknown' for p in preds]
    correct = [1 if p == gt else 0 for p, gt in zip(preds, label_strs)]
    accuracy = np.sum(np.array(correct)[known_mask]) / max(np.sum(known_mask), 1)
    print(f"EVM Accuracy (known only): {accuracy:.4f}  (threshold={evm_threshold})")
    return accuracy, preds

# >>>>>>>>>>>>>>>>>>>> Main <<<<<<<<<<<<<<<<<<<<<<<

def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    all_classes = [c.strip() for c in args.all_classes.split(',')]
    num_classes = len(all_classes)
    base_num_classes = 16

    rvl_cdip_class= ['letter', 'form', 'email', 'handwritten', 'advertisement', 'scientific_report', 'scientific_publication', 'specification', 'file_folder', 'news_article', 'budget', 'invoice', 'presentation', 'questionnaire', 'resume', 'memo']
    tobacco_class= ['letter', 'form', 'email', 'advertisement', 'scientific_report', 'news_article', 'resume', 'memo', 'Note', 'Report']


    print("Loading Base Model.....")
    base_model = LayoutLMv3(
        text_model_name='bert-base-uncased',
        vision_model_name='vit_base_patch16_224',
        num_labels=base_num_classes
    ).to(device)
    base_model.device = device # For EVM routines
    print("Basen Model Loading from :", args.base_model_path)
    checkpoint = torch.load(args.base_model_path, map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    expand_classifier(base_model, base_num_classes, num_classes, device)
    model = base_model
    model.device = device # For EVM routines
    print("Loading Base Model..... Complete!")
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    if args.training_mode == 'last_layer':
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    bias_correction = BiasCorrectionLayer(num_classes, 0).to(device)

    print(f"Loading base domain ({args.dataset_base}) train data for exemplar selection .....")
    base_data_dir = os.path.join(args.data_dir, 'all_prepdataset')
    base_train_loader = get_incremental_dataloader(
        dataset_name=args.dataset_base,
        ocr_tensor_file=args.ocr_tensor_path_base,
        classes=rvl_cdip_class,
        image_dir=base_data_dir,
        split="train",
        batch_size=args.batch_size,
        images_per_class=args.max_exemplars,
        seed=args.seed,
    )
    print(f"Base domain training samples: {len(base_train_loader.dataset)}")


    print(f"Loading incremental domain ({args.dataset_inc}) train data .....")
    inc_data_dir = os.path.join(args.data_dir, 'Tobacco3482-jpg')
    inc_train_loader = get_incremental_dataloader(
        dataset_name=args.dataset_inc,
        ocr_tensor_file=args.ocr_tensor_path_inc,
        classes=tobacco_class,
        image_dir=inc_data_dir,
        split="train",
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"Incremental domain training samples (limited): {len(inc_train_loader.dataset)}")


    print(f"Loading base domain ({args.dataset_base}) val data .....")
    base_val_loader = get_incremental_dataloader(
        dataset_name=args.dataset_base,
        ocr_tensor_file=args.ocr_tensor_path_base,
        classes=all_classes,
        image_dir=base_data_dir,
        split="val",
        batch_size=args.batch_size,
        images_per_class=1250,
        seed=args.seed,
    )
    print(f"Base domain val samples: {len(base_val_loader.dataset)}")


    print(f"Loading incremental domain ({args.dataset_inc}) val data .....")
    inc_val_loader = get_incremental_dataloader(
        dataset_name=args.dataset_inc,
        ocr_tensor_file=args.ocr_tensor_path_inc,
        classes=all_classes,
        image_dir=inc_data_dir,
        split="val",
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"Incremental domain val samples: {len(inc_val_loader.dataset)}")


    print(f"Loading base domain ({args.dataset_base}) test data .....")
    base_test_loader = get_incremental_dataloader(
        dataset_name=args.dataset_base,
        ocr_tensor_file=args.ocr_tensor_path_base,
        classes=all_classes,
        image_dir=base_data_dir,
        split="test",
        batch_size=args.batch_size,
        images_per_class=1250,
        seed=args.seed,
    )
    print(f"Base domain test samples: {len(base_test_loader.dataset)}")


    print(f"Loading incremental domain ({args.dataset_inc}) test data .....")
    inc_test_loader = get_incremental_dataloader(
        dataset_name=args.dataset_inc,
        ocr_tensor_file=args.ocr_tensor_path_inc,
        classes=all_classes,
        image_dir=inc_data_dir,
        split="test",
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"Incremental domain test samples: {len(inc_test_loader.dataset)}")

    # Prepare exemplars from base domain for replay
    print("Preparing exemplars from base domain for replay ...")
    base_class_indices = [all_classes.index(c) for c in all_classes]
    exemplar_handler = ExemplarHandler(max_exemplars_per_class=args.max_exemplars, selection_method=args.exemplar_selection)
    if args.exemplar_selection == 'herding':
        exemplar_handler.update_exemplars(base_train_loader.dataset, base_class_indices, model, device)
    else:
        exemplar_handler.update_exemplars(base_train_loader.dataset, all_classes, model, device)
    exemplar_samples = exemplar_handler.get_exemplar_dataset()

    # Combine incremental domain dataset and base domain exemplars for training
    combined_dataset = ConcatDataset([inc_train_loader.dataset, exemplar_samples])
    combined_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print("Exemplar dataset prepared.")

    print(f"Total exemplar samples: {len(exemplar_samples)}")
    print(f"Total incremental domain train samples: {len(inc_train_loader.dataset)}")
    print(f"Combined training dataset samples: {len(combined_dataset)}")

    ewc = EWC(model, base_train_loader, device, fisher_n=500, lambda_ewc=args.lambda_ewc) if args.use_ewc else None
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.01)

    start_epoch = 0
    best_val_acc = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    if args.resume and args.resume_checkpoint:
        print(f"Resuming training from checkpoint {args.resume_checkpoint} ...")
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        bias_correction.load_state_dict(ckpt['bias_correction_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)

    print("Training started...")
    scaler = amp.GradScaler()

    # >>>>>>> EVM Initialization <<<<<<<
    print("Initializing EVM...")
    feature_dict, _, _ = extract_features_for_evm(model, inc_train_loader, device, all_classes)
    #evm_hybrid = EVMClassifier(tailsize=args.evm_tailsize, cover_threshold=args.evm_threshold)
    evm_hybrid = IncrementalEVM(tailsize=0.5, ev_budget=10, cover_threshold=0.7)
    evm_hybrid.fit(feature_dict)
    print("EVM Initialized.")

    if args.training_mode == 'last_layer':
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    for epoch in range(start_epoch, args.num_epochs):
        gc.collect()
        torch.cuda.empty_cache()

        model.train()
        running_loss = 0
        correct = 0
        total = 0
        loop = tqdm(combined_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", leave=False)

        batch_features, batch_labels = [], []

        for i, batch in enumerate(loop):
            for k in ['input_ids', 'attention_mask', 'bbox', 'pixel_values']:
                batch[k] = batch[k].to(device)
            batch['labels'] = torch.tensor(
                [all_classes.index(lbl) if isinstance(lbl, str) else lbl for lbl in batch['labels']],
                device=device
            )
            optimizer.zero_grad()

            with amp.autocast():
                inputs = {k: v for k, v in batch.items() if k != 'labels'}
                with torch.no_grad():
                    old_logits = base_model(**inputs)
                # Forward and feature extraction
                new_logits_raw = model(**inputs)
                # Features for EVM (assumes 'extract_features' present in your LayoutLMv3)
                feats = model.extract_features(
                    batch['input_ids'], batch['bbox'], batch['attention_mask'], batch['pixel_values']
                )

                # Hybrid loss (CE + KD + EVM)
                cls_loss = F.cross_entropy(new_logits_raw, batch['labels'])
                kd_loss = distillation_loss(new_logits_raw, old_logits)
                evm_loss_total = hybrid_loss(new_logits_raw, feats, batch['labels'], evm_hybrid, criterion, all_classes, args.lambda_evm)
                loss = cls_loss + kd_loss + (evm_loss_total - cls_loss)  # Add only evm part (since hybrid_loss also includes CE)
                if ewc:
                    loss += ewc.penalty(model)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * batch['labels'].size(0)
            _, predicted = new_logits_raw.max(1)
            total += batch['labels'].size(0)
            correct += predicted.eq(batch['labels']).sum().item()

            batch_features.append(feats.cpu())
            batch_labels.append(batch['labels'].cpu())

            # Cleanup
            del batch, old_logits, new_logits_raw, loss, cls_loss, kd_loss, predicted, feats
            torch.cuda.empty_cache()
            gc.collect()

            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            train_acc = correct / total if total > 0 else 0
            train_loss = running_loss / total if total > 0 else 0
            loop.set_postfix(loss=train_loss, acc=train_acc)

        # Update EVM with features from epoch
        features_epoch = torch.cat(batch_features, 0)
        labels_epoch = torch.cat(batch_labels, 0).numpy()
        feature_dict_epoch = {}
        for i, l in enumerate(labels_epoch):
            class_name = all_classes[l] if isinstance(all_classes[0], str) else l
            feature_dict_epoch.setdefault(class_name, []).append(features_epoch[i].numpy())
        for k in feature_dict_epoch:
            feature_dict_epoch[k] = np.stack(feature_dict_epoch[k], axis=0)
        evm_hybrid.incremental_update(feature_dict_epoch)

        print(f"Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}")

        # Bias correction update after increment training
        model.eval()
        with torch.no_grad():
            all_biases = []
            for batch in inc_train_loader:
                for k in ['input_ids', 'attention_mask', 'bbox', 'pixel_values']:
                    batch[k] = batch[k].to(device)
                inputs = {k: v for k, v in batch.items() if k != 'labels'}
                logits = model(**inputs)
                all_biases.append(logits.mean(dim=0, keepdim=True))
            if all_biases:
                mean_bias = torch.cat(all_biases, dim=0).mean(dim=0, keepdim=True)
                bias_correction.bias_vector.data = mean_bias.data.clone().detach()

        # Evaluate on both base and incremental domain test data
        val_loss_base, val_acc_base, val_p_base, val_r_base, val_f1_base, val_gil_base, val_class_acc_base = evaluate(
            model, base_val_loader, device, all_classes,
            args.full_model_acc,
            split_name="RVL-CDIP Test",
            bias_correction=bias_correction
        )
        val_loss_inc, val_acc_inc, val_p_inc, val_r_inc, val_f1_inc, val_gil_inc, val_class_acc_inc = evaluate(
            model, inc_val_loader, device, all_classes,
            args.full_model_acc,
            split_name="Tobacco-3482 Test",
            bias_correction=bias_correction
        )
        print(f"\nEpoch {epoch + 1}: RVL-CDIP Val Acc {val_acc_base:.4f}, Tobacco-3482 Val Acc {val_acc_inc:.4f}")

        # Save best model based on combined accuracies or your preferred metric
        combined_acc = (val_acc_base + val_acc_inc) / 2
        if combined_acc > best_val_acc:
            best_val_acc = combined_acc
            save_path = os.path.join(args.checkpoint_dir, f"layoutlmv3_domain_incremental_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'bias_correction_state_dict': bias_correction.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, save_path)
            print(f"Saved best model checkpoint: {save_path}")

    print("Training completed. Loading best model for testing ...")
    best_ckpt = torch.load(os.path.join(args.checkpoint_dir, f"layoutlmv3_domain_incremental_best.pt"), map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()

    print("Evaluating on both domains test data ...")
    test_loss_base, test_acc_base, test_p_base, test_r_base, test_f1_base, test_gil_base, test_class_acc_base = evaluate(
        model, base_test_loader, device, all_classes, args.full_model_acc, split_name="RVL-CDIP Test"
    )
    test_loss_inc, test_acc_inc, test_p_inc, test_r_inc, test_f1_inc, test_gil_inc, test_class_acc_inc = evaluate(
        model, inc_test_loader, device, all_classes, args.full_model_acc, split_name="Tobacco-3482 Test"
    )
    print(f"Final RVL-CDIP Test Loss: {test_loss_base:.4f}, Accuracy: {test_acc_base:.4f}, F1: {test_f1_base:.4f}")
    print(f"Final Tobacco-3482 Test Loss: {test_loss_inc:.4f}, Accuracy: {test_acc_inc:.4f}, F1: {test_f1_inc:.4f}")

    # >>>>> Final EVM eval <<<<<
    print("Final EVM evaluation on test datasets:")
    for domain_name, loader in [("RVL-CDIP", base_test_loader), ("Tobacco-3482", inc_test_loader)]:
        accuracy, _ = evm_evaluate(model, loader, all_classes, args.evm_tailsize, args.evm_threshold)
        print(f"Final EVM accuracy on domain '{domain_name}': {accuracy:.4f}")

if __name__ == "__main__":
    main()
