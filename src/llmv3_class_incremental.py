import argparse
import os
import torch
import torch.nn.functional as F
import gc
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

def parse_args():
    parser = argparse.ArgumentParser(description="LayoutLMv3 Incremental Training")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ocr_tensor_path', required=True)
    parser.add_argument('--all_classes', required=True)
    parser.add_argument('--base_classes', required=True)
    parser.add_argument('--unseen_classes', required=True)
    parser.add_argument('--dataset_name', default='rvl_cdip', choices=['rvl_cdip', 'tobacco3482'])
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
    parser.add_argument('--images_per_class', type=int, default=12500, help="Max images per unseen class")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--test_classes', default=None, help="Comma separated test classes if different")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    all_classes = [c.strip() for c in args.all_classes.split(',')]
    base_classes = [c.strip() for c in args.base_classes.split(',')]
    unseen_classes = [c.strip() for c in args.unseen_classes.split(',')]

    if args.test_classes:
        test_classes = [c.strip() for c in args.test_classes.split(',')]
    else:
        test_classes = base_classes + unseen_classes

    base_num_classes = len(base_classes)
    total_num_classes = len(all_classes)

    base_model_acc = 0.9633
    print("Loading Base Model.....")
    base_model = LayoutLMv3(
        text_model_name='bert-base-uncased',
        vision_model_name='vit_base_patch16_224',
        num_labels=base_num_classes
    ).to(device)
    checkpoint = torch.load(args.base_model_path, map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    expand_classifier(base_model, base_num_classes, base_num_classes+1, device)
    model = base_model
    print("Loading Base Model..... Complete!")
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    if args.training_mode == 'last_layer':
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    bias_correction = BiasCorrectionLayer(base_num_classes, len(unseen_classes)).to(device)

    print("Loading base train data for exemplar selection .....")
    base_train_loader = get_incremental_dataloader(
        dataset_name=args.dataset_name,
        ocr_tensor_file=args.ocr_tensor_path,
        classes=base_classes,
        image_dir=args.data_dir,
        split="train", 
        batch_size=args.batch_size,
        images_per_class=args.max_exemplars,
        seed=args.seed,
    )
    print(f"Base training samples: {len(base_train_loader.dataset)}")

    print("Loading unseen train data .....")
    unseen_train_loader = get_incremental_dataloader(
        dataset_name=args.dataset_name,
        ocr_tensor_file=args.ocr_tensor_path,
        classes=unseen_classes,
        image_dir=args.data_dir,
        split="train", 
        batch_size=args.batch_size,
        images_per_class=args.images_per_class,
        seed=args.seed,
    )
    print(f"Unseen class samples (limited): {len(unseen_train_loader.dataset)}")

    val_test_img = 1250
    print("Loading validation data .....")
    val_loader = get_incremental_dataloader(
        dataset_name=args.dataset_name,
        ocr_tensor_file=args.ocr_tensor_path,
        classes=base_classes + unseen_classes,
        image_dir=args.data_dir,
        split="val",
        batch_size=args.batch_size,
        images_per_class=val_test_img,
        seed=args.seed,
    )
    print(f"Validation samples: {len(val_loader.dataset)}")

    print("Loading test data .....")
    test_loader = get_incremental_dataloader(
        dataset_name=args.dataset_name,
        ocr_tensor_file=args.ocr_tensor_path,
        classes=base_classes + unseen_classes,
        image_dir=args.data_dir,
        split="test",
        batch_size=args.batch_size,
        images_per_class=val_test_img,
        seed=args.seed,
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    print("Preparing exemplars from base classes ...")
    base_class_indices = [all_classes.index(c) for c in base_classes]
    exemplar_handler = ExemplarHandler(max_exemplars_per_class=args.max_exemplars, selection_method=args.exemplar_selection)
    if args.exemplar_selection == 'herding':
        #exemplar_handler.update_exemplars(base_train_loader.dataset, base_classes, model, device)
        exemplar_handler.update_exemplars(base_train_loader.dataset, base_class_indices, model, device)
    else:
        exemplar_handler.update_exemplars(base_train_loader.dataset, base_classes, model, device)
    exemplar_samples = exemplar_handler.get_exemplar_dataset()

    combined_dataset = ConcatDataset([unseen_train_loader.dataset, exemplar_samples])
    combined_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print("Exemplar dataset prepared.")

    """
    print("Checking exemplars per base class:")
    exemplar_counts = {}
    for cls in base_class_indices:
        count = sum(1 for ex in exemplar_samples if (isinstance(ex['labels'], int) and ex['labels'] == cls) or 
                                                (isinstance(ex['labels'], str) and all_classes[cls] == ex['labels']))
        exemplar_counts[cls] = count
        print(f"  Class {all_classes[cls]} (index {cls}): {count} exemplars")
    """
    print(f"Total exemplar samples: {len(exemplar_samples)}")
    print(f"Total unseen train samples: {len(unseen_train_loader.dataset)}")
    print(f"Combined training dataset samples: {len(combined_dataset)}")
    


    ewc = EWC(model, base_train_loader, device, fisher_n=500, lambda_ewc=args.lambda_ewc) if args.use_ewc else None

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.01)

    start_epoch = 0
    best_val_acc = 0.0
    patience_counter = 0

    if args.resume and args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        bias_correction.load_state_dict(ckpt['bias_correction_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        patience_counter = ckpt.get('patience_counter', 0)

    print("Training started...")

    # Initialize GradScaler for mixed precision
    scaler = amp.GradScaler()

    # Freeze all layers except last classifier layer as before
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

        for i, batch in enumerate(loop):
            for k in ['input_ids', 'attention_mask', 'bbox', 'pixel_values']:
                batch[k] = batch[k].to(device)

            #batch['labels'] = torch.tensor(
            #    [all_classes.index(lbl) if isinstance(lbl, str) else lbl for lbl in batch['labels']],
            #    device=device
            #)
            batch['labels'] = torch.tensor(
                [all_classes.index(lbl) if isinstance(lbl, str) else int(lbl) for lbl in batch['labels']],
                dtype=torch.long,
                device=device
            )


            optimizer.zero_grad()

            with amp.autocast():
                inputs = {k: v for k, v in batch.items() if k != 'labels'}
                with torch.no_grad():
                    old_logits = base_model(**inputs)

                new_logits_raw = model(**inputs)

                cls_loss = F.cross_entropy(new_logits_raw, batch['labels'])
                #kd_loss = distillation_loss(new_logits_raw, old_logits)
                #loss = cls_loss + kd_loss
                loss = cls_loss
                if ewc:
                    loss += ewc.penalty(model)


            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * batch['labels'].size(0)
            _, predicted = new_logits_raw.max(1)
            total += batch['labels'].size(0)
            correct += predicted.eq(batch['labels']).sum().item()

            # Aggressive cleanup
            del batch, old_logits, new_logits_raw, loss, cls_loss, predicted #kd_loss
            torch.cuda.empty_cache()
            gc.collect()

            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            train_acc = correct / total if total > 0 else 0
            train_loss = running_loss / total if total > 0 else 0
            loop.set_postfix(loss=train_loss, acc=train_acc)

        print(f"\nEpoch {epoch + 1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}")

        # Bias correction update *after* unseen class incremental step
        model.eval()
        with torch.no_grad():
            all_biases = []
            for batch in unseen_train_loader:
                for k in ['input_ids', 'attention_mask', 'bbox', 'pixel_values']:
                    batch[k] = batch[k].to(device)
                inputs = {k: v for k, v in batch.items() if k != 'labels'}
                logits = model(**inputs)
                all_biases.append(logits.mean(dim=0, keepdim=True))
            if all_biases:
                mean_bias = torch.cat(all_biases, dim=0).mean(dim=0, keepdim=True)
                bias_correction.bias_vector.data = mean_bias.data.clone().detach()

        # Validation with bias correction applied
        val_loss, val_acc, p, r, f1, gil, class_acc = evaluate(
            model, val_loader, device, all_classes,
            args.full_model_acc,
            split_name="Val",
            bias_correction=bias_correction
        )
        print(f"\nEpoch {epoch + 1}: Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

        gil_previous = (val_acc - args.full_model_acc) / (1 - args.full_model_acc)
        print(f"\nGIL_PreClass-val:{gil_previous:.4f}")

        gil_base = (val_acc - base_model_acc) / (1 - base_model_acc)
        print(f"\nGIL_Base-val:{gil_base:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_path = os.path.join(args.checkpoint_dir, f"layoutlmv3_cil_incremental_{args.unseen_classes}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'bias_correction_state_dict': bias_correction.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'patience_counter': patience_counter,
            }, save_path)
            print(f"Saved best model checkpoint: {save_path}")
        else:
            patience_counter += 1

        print(f"Patience counter: {patience_counter} / {args.patience}")
        if patience_counter > args.patience:
            print(f"Early stopping triggered. Patience counter exceeded {args.patience}.")
            break
    print("Training completed. Loading best model for testing ...")
    best_ckpt = torch.load(os.path.join(args.checkpoint_dir, f"layoutlmv3_cil_incremental_{args.unseen_classes}_best.pt"), map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()

    test_loss, test_acc, test_p, test_r, test_f1, test_gil, test_class_acc = evaluate(model, test_loader, device, all_classes, args.full_model_acc, split_name="Test")

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, GIL_Previous:{test_gil:.4f}")

    gil_base = (test_acc - base_model_acc) / (1 - base_model_acc)
    print(f"\nGIL_Base-Test:{gil_base:.4f}")

if __name__ == "__main__":
    main()
