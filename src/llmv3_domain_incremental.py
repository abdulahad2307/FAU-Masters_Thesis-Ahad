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
    parser = argparse.ArgumentParser(description="LayoutLMv3 Domain-Incremental Training")
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
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    all_classes = [c.strip() for c in args.all_classes.split(',')]
    num_classes = len(all_classes)

    rvl_cdip_class= ['letter', 'form', 'email', 'handwritten', 'advertisement', 'scientific_report', 'scientific_publication', 'specification', 'file_folder', 'news_article', 'budget', 'invoice', 'presentation', 'questionnaire', 'resume', 'memo']
    tobacco_class= ['letter', 'form', 'email', 'advertisement', 'scientific_report', 'news_article', 'resume', 'memo', 'Note', 'Report']


    print("Loading Base Model.....")
    base_num_classes = 16
    base_model = LayoutLMv3(
        text_model_name='bert-base-uncased',
        vision_model_name='vit_base_patch16_224',
        num_labels=base_num_classes
    ).to(device)
    print("Basen Model Loading fronm :", args.base_model_path)
    checkpoint = torch.load(args.base_model_path, map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    expand_classifier(base_model, base_num_classes, num_classes, device)
    model = base_model
    print("Loading Base Model..... Complete!")
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    if args.training_mode == 'last_layer':
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    bias_correction = BiasCorrectionLayer(num_classes, 0).to(device)  # No class expansion here

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

    # Freeze all layers except last classifier layer if specified
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

            batch['labels'] = torch.tensor(
                [all_classes.index(lbl) if isinstance(lbl, str) else lbl for lbl in batch['labels']],
                device=device
            )

            optimizer.zero_grad()

            with amp.autocast():
                inputs = {k: v for k, v in batch.items() if k != 'labels'}
                # Use base_model for distillation with no grad
                with torch.no_grad():
                    old_logits = base_model(**inputs)

                new_logits_raw = model(**inputs)

                cls_loss = F.cross_entropy(new_logits_raw, batch['labels'])
                kd_loss = distillation_loss(new_logits_raw, old_logits)
                loss = cls_loss + kd_loss
                if ewc:
                    loss += ewc.penalty(model)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * batch['labels'].size(0)
            _, predicted = new_logits_raw.max(1)
            total += batch['labels'].size(0)
            correct += predicted.eq(batch['labels']).sum().item()

            # Cleanup
            del batch, old_logits, new_logits_raw, loss, cls_loss, kd_loss, predicted
            torch.cuda.empty_cache()
            gc.collect()

            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            train_acc = correct / total if total > 0 else 0
            train_loss = running_loss / total if total > 0 else 0
            loop.set_postfix(loss=train_loss, acc=train_acc)

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
                #logits = model(**batch).logits
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
        print(f"Epoch {epoch + 1}: RVL-CDIP Test Acc {val_acc_base:.4f}, Tobacco-3482 Test Acc {val_acc_inc:.4f}")

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
        model, base_test_loader, device, all_classes, args.full_model_acc, split_name="Test"
    )
    test_loss_inc, test_acc_inc, test_p_inc, test_r_inc, test_f1_inc, test_gil_inc, test_class_acc_inc = evaluate(
        model, inc_test_loader, device, all_classes, args.full_model_acc, split_name="Test"
    )
    print(f"Final RVL-CDIP Test Loss: {test_loss_base:.4f}, Accuracy: {test_acc_base:.4f}, F1: {test_f1_base:.4f}, G_IL-previous={test_gil_base:.4f}")
    print(f"Final Tobacco-3482 Test Loss: {test_loss_inc:.4f}, Accuracy: {test_acc_inc:.4f}, F1: {test_f1_inc:.4f}, G_IL-previous={test_gil_inc:.4f}")


if __name__ == "__main__":
    main()
