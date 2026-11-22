import torch
import numpy as np
import os

from .DyCalKT import DyCalKT
# from .dkt import DKT
# from .akt import AKT
# from .ukt import UKT

# from .folibikt import folibiKT

device = "cpu" if not torch.cuda.is_available() else "cuda"


def init_model(model_name, model_config, data_config, emb_type):
    if model_name in ["dkt", "DKT"]:
        model = DKT(
            data_config["num_c"],
            **model_config,
            emb_type=emb_type,
            emb_path=data_config["emb_path"]
        ).to(device)

    elif model_name in ["sakt", "SAKT"]:
        model = SAKT(
            data_config["num_c"],
            **model_config,
            emb_type=emb_type,
            emb_path=data_config["emb_path"]
        ).to(device)

    elif model_name in ["saint", "SAINT"]:
        model = SAINT(
            data_config["num_q"],
            data_config["num_c"],
            **model_config,
            emb_type=emb_type,
            emb_path=data_config["emb_path"]
        ).to(device)

    elif model_name in ["akt", "AKT"]:
        print("[DEBUG-init_model] enter AKT branch, model_name =", repr(model_name))

        akt_allowed_keys = [
            "d_model",
            "n_blocks",
            "dropout",
            "d_ff",
            "kq_same",
            "final_fc_dim",
            "num_attn_heads",
            "separate_qa",
            "l2",
            "pretrain_dim",
            "num_diff",
            "num_cluster",
            "num_time",
            "num_behavior",
            "use_dynamic_modal",
            "use_anchor_alignment",
            "anchor_weight",
        ]
        akt_config = {k: v for k, v in model_config.items() if k in akt_allowed_keys}
        print("[DEBUG-init_model] akt_config keys =", sorted(akt_config.keys()))

        model = AKT(
            data_config["num_c"],
            data_config["num_q"],
            **akt_config,
            emb_type=emb_type,
            emb_path=data_config["emb_path"]
        ).to(device)
    elif model_name in ["ukt", "UKT"]:
        print("[DEBUG-init_model] enter UKT branch, model_name =", repr(model_name))

        ukt_allowed_keys = [
            "d_model",
            "n_blocks",
            "dropout",
            "d_ff",
            "loss1",
            "loss2",
            "loss3",
            "start",
            "num_layers",
            "nheads",
            "seq_len",
            "kq_same",
            "final_fc_dim",
            "final_fc_dim2",
            "num_attn_heads",
            "separate_qa",
            "use_CL",
            "cl_weight",
            "use_uncertainty_aug",
            "l2",
            "atten_type",
            "pretrain_dim",
            "num_diff",
            "num_cluster",
            "num_time",
            "num_behavior",
            "use_dynamic_modal",
            "use_anchor_alignment",
            "anchor_weight",
        ]
        ukt_config = {k: v for k, v in model_config.items() if k in ukt_allowed_keys}
        print("[DEBUG-init_model] ukt_config keys =", sorted(ukt_config.keys()))
        model = UKT(
            data_config["num_c"],
            data_config["num_q"],
            **ukt_config,
            emb_type=emb_type,
            emb_path=data_config["emb_path"]
        ).to(device)

        

    elif model_name == "iekt":
        model = IEKT(
            num_q=data_config["num_q"],
            num_c=data_config["num_c"],
            max_concepts=data_config["max_concepts"],
            **model_config,
            emb_type=emb_type,
            emb_path=data_config["emb_path"],
            device=device
        ).to(device)

    elif model_name == "simplekt":
        model = simpleKT(
            data_config["num_c"],
            data_config["num_q"],
            **model_config,
            emb_type=emb_type,
            emb_path=data_config["emb_path"]
        ).to(device)

    elif model_name in ["DyCalKT", "dycalkt"]:
        model = DyCalKT(
            data_config["num_c"],
            data_config["num_q"],
            **model_config,
            emb_type=emb_type,
            emb_path=data_config["emb_path"]
        ).to(device)

    else:
        raise ValueError(f"Unsupported model_name in init_model: {repr(model_name)}")

    return model

def load_model(model_name, model_config, data_config, emb_type, ckpt_path):
    model = init_model(model_name, model_config, data_config, emb_type)
    net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
    model.load_state_dict(net)
    return model
