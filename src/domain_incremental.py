import argparse
import torch
import os
import time
from torch import nn

from utils.domain_IL.dil_train_utils import (
    train_one_epoch_dil,
    evaluate_dil,
    save_checkpoint_dil,
    load_checkpoint_dil,
    set_finetune_mode
)

from utils.dataloader import EAML_DataLoader, DataLoader as DocFormerLoader

from utils.eaml.eaml_model import EAMLModel
from utils.docformer.model import DocFormer
from utils.docformer.config import DocFormerConfig

def main(args):
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ## ---------------------- Loading Class Info ------------------------------------- ##

    if args.class_list_path:
        import json
        with open(args.class_list_path) as f:
            class_list = json.load(f)
    else:
        class_list = None  # fallback to auto-detect

    # ---------------------------------- Loading Model(s)--------------------------------- ##

    models = {}
    if args.model in ["eaml", "both"]:
        model_eaml = EAMLModel(num_classes=args.num_classes).to(device)
        model_eaml.load_state_dict(torch.load(args.eaml_path, map_location=device))
        set_finetune_mode(model_eaml, mode=args.finetune_mode, encoder_unfreeze_depth=args.unfreeze_depth)
        models["eaml"] = model_eaml

    if args.model in ["docformer", "both"]:
        config = DocFormerConfig()
        model_docformer = DocFormer(config,num_classes=args.num_classes).to(device)
        model_docformer.load_state_dict(torch.load(args.docformer_path, map_location=device))
        set_finetune_mode(model_docformer, mode=args.finetune_mode, encoder_unfreeze_depth=args.unfreeze_depth)
        models["docformer"] = model_docformer

    ## Wrapping models if ensemble requested
    if args.ensemble_first and args.model == "both":
        model = models
    else:
        model = list(models.values())[0]  # single model case

    ## ----------------------------- LoadingData --------------------------------------- ##

    data_start = time.time()
    if args.model == "eaml" or (args.model == "both" and args.ensemble_first):
        dataloader = EAML_DataLoader(data_dir=args.data_dir, batch_size=args.batch_size, class_list=class_list)
    else:
        dataloader = DocFormerLoader(data_dir=args.data_dir, batch_size=args.batch_size)

    train_loader = dataloader.get_loader("train")
    val_loader = dataloader.get_loader("val")
    print(f"Loaded data in {time.time() - data_start:.2f}s")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters() if not isinstance(model, dict) else sum((list(m.parameters()) for m in model.values()), [])),
                                 lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = os.path.join(args.ckpt_dir, f"{args.model}_dil.pth")
    start_epoch = load_checkpoint_dil(model, optimizer, checkpoint_path, device)

    ## -------------------- Training Loop ------------ ##
    for epoch in range(start_epoch, args.epochs):
        print(f"\n===Epoch {epoch+1}/{args.epochs} ===")
        epoch_start = time.time()

        train_one_epoch_dil(model, train_loader, optimizer, criterion, device)
        acc = evaluate_dil(model, val_loader, device)

        save_checkpoint_dil(model, optimizer, epoch, checkpoint_path)
        print(f"Epoch time: {time.time() - epoch_start:.2f}s")

    print(f"All done in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain Incremental Learning")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints_dil")
    parser.add_argument("--eaml_path", type=str, default="./eaml.pth")
    parser.add_argument("--docformer_path", type=str, default="./docformer.pth")
    parser.add_argument("--model", choices=["eaml", "docformer", "both"], default="both")
    parser.add_argument("--ensemble_first", action="store_true", help="Do ensemble before DIL")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--finetune_mode", choices=["head_only", "partial_finetune", "full_finetune"], default="partial_finetune")
    parser.add_argument("--unfreeze_depth", type=int, default=2)
    parser.add_argument("--class_list_path", type=str, help="Optional path to JSON file with class list")
    parser.add_argument("--num_classes", type=int, default=16)

    args = parser.parse_args()
    main(args)
