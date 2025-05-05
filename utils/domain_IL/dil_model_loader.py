import torch
from utils.domain_IL.dil_model_wrapper import DomainIncrementalWrapper
from utils.ensemble.joint_wrapper import JointModelWrapper

from utils.eaml.eaml_model import EAMLModel
from utils.docformer.model import DocFormer
from utils.docformer.config import DocFormerConfig

def load_eaml(checkpoint_path, num_classes):
    model = EAMLModel(num_classes=num_classes)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

def load_docformer(checkpoint_path, num_classes):
    config = DocFormerConfig()
    model =DocFormer(config,num_classes=num_classes)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

def prepare_dil_models(config, class_per_domain: dict, device='cuda'):
    """
    Returns models for DIL setup.
    
    config: {
        'strategy': 'eaml_only' | 'docformer_only' | 'pre_ensemble' | 'post_ensemble',
        'eaml_ckpt': 'path/to/eaml_best.pth',
        'doc_ckpt': 'path/to/docformer_best.pth',
        'dil_mode': True/False
    }
    """

    strategy = config.get("strategy")
    dil = config.get("dil_mode", True)

    ## === Loading Base Models === ##
    eaml = load_eaml(config["eaml_ckpt"], num_classes=sum(class_per_domain.values())).to(device)
    docformer = load_docformer(config["doc_ckpt"], num_classes=sum(class_per_domain.values())).to(device)

    ## === Preparing according to strategy === ##
    if strategy == "eaml_only":
        model = eaml
    elif strategy == "docformer_only":
        model = docformer
    elif strategy == "pre_ensemble":
        model = JointModelWrapper(eaml, docformer, mode="average")  # or mode="concat"
    elif strategy == "post_ensemble":
        model = {
            "eaml": eaml,
            "docformer": docformer
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    ## === Wrapping in DomainIncrementalWrapper if required === ##
    if dil:
        if isinstance(model, dict):  # post-ensemble, wrapping each model separately
            model = {
                name: DomainIncrementalWrapper(m, class_per_domain).to(device)
                for name, m in model.items()
            }
        else:
            model = DomainIncrementalWrapper(model, class_per_domain).to(device)

    return model
