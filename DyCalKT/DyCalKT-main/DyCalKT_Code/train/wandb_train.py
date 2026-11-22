import os
import argparse
import json
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
torch.set_num_threads(4) 
from torch.optim import SGD, Adam
import copy
import sys
sys.path.append("../")
from models import train_model,init_model
from utils.utils import debug_print,set_seed
from datasets import init_dataset4train
import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

def save_config(train_config, model_config, data_config, params, save_dir):
    d = {"train_config": train_config, 'model_config': model_config, "data_config": data_config, "params": params}
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout)

def resolve_modal_dims(dataset_name, data_config, params):
    dcfg = data_config[dataset_name]

    dims = {
        "num_skill": max(params.get("num_skill", 0), dcfg.get("num_c", 0)),
        "num_diff": int(dcfg.get("num_diff", 0)),
        "num_time": int(dcfg.get("num_time", 0)),
        "num_cluster": int(dcfg.get("num_cluster", 0)),
        "num_behavior": int(dcfg.get("num_behavior", 0)),
    }

    # ===== 消融开关：只关闭当前数据集已存在的模态，不影响其它模态 =====
    if int(params.get("ablate_diff", 0)) == 1:
        dims["num_diff"] = 0

    if int(params.get("ablate_behavior", 0)) == 1:
        dims["num_behavior"] = 0

    if int(params.get("ablate_cluster", 0)) == 1:
        dims["num_cluster"] = 0

    if int(params.get("ablate_time", 0)) == 1:
        dims["num_time"] = 0

    return dims


def main(params):
    set_seed(params["seed"])
    model_name, dataset_name, fold, emb_type, save_dir = params["model_name"], params["dataset_name"], \
        params["fold"], params["emb_type"], params["save_dir"]

    debug_print(text = "load config files.",fuc_name="main")
    

    with open("../configs/kt_config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]
        # if model_name in ["dkvmn","deep_irt", "sakt", "saint","saint++", "akt","folibikt", "atkt", "lpkt", "skvmn", "dimkt"]:
        if model_name in ["dkvmn","deep_irt", "sakt", "saint","saint++", "akt","folibikt", "atkt", "lpkt", "skvmn", "dimkt"]:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["simplekt", "bakt_time", "sparsekt"]:
            train_config["batch_size"] = 128 ## because of OOM
        if model_name in ["DyCalKT", "dycalkt", "dkt", "DKT"]:
            train_config["batch_size"] = 64
        if model_name in ["gkt"]:
            train_config["batch_size"] = 16
        if model_name in ["qdkt", "qikt"] and dataset_name in ['algebra2005', 'bridge2algebra2006']:
            train_config["batch_size"] = 32
        if model_name in ["dtransformer"]:
            train_config["batch_size"] = 32 ## because of OOM
        model_config = copy.deepcopy(params)

        # ===== DyCalKT 消融实验参数 =====
        if model_name in ["DyCalKT", "dycalkt"]:
            model_config["ablation_mode"] = params.get(
                "ablation_mode",
                "full"
            )

            model_config["use_cal_loss"] = bool(
                int(params.get("use_cal_loss", 0))
            )

            model_config["use_con_loss"] = bool(
                int(params.get("use_con_loss", 0))
            )


        
        for key in [
            "model_name", "dataset_name", "emb_type", "save_dir", "fold", "seed",
            "ablate_diff", "ablate_behavior", "ablate_cluster", "ablate_time",
            "num_epochs", "batch_size"
        ]:
            model_config.pop(key, None)
        if 'batch_size' in params:
            train_config["batch_size"] = params['batch_size']
        if 'num_epochs' in params:
            train_config["num_epochs"] = params['num_epochs']
        # model_config = {"d_model": params["d_model"], "n_blocks": params["n_blocks"], "dropout": params["dropout"], "d_ff": params["d_ff"]}
    batch_size, num_epochs, optimizer = train_config["batch_size"], train_config["num_epochs"], train_config["optimizer"]

    with open("../configs/data_config.json") as fin:
        data_config = json.load(fin)
    if 'maxlen' in data_config[dataset_name]:#prefer to use the maxlen in data config
        train_config["seq_len"] = data_config[dataset_name]['maxlen']
    seq_len = train_config["seq_len"]

    if model_name in ["DyCalKT", "dycalkt", "dkt", "DKT", "akt", "AKT", "ukt", "UKT"]:

        model_config["atten_type"]=params["atten_type"]
        # ===== 新增：把动态模态相关参数真正传进模型 =====
        model_config["use_dynamic_modal"] = bool(int(params["use_dynamic_modal"]))
        model_config["use_anchor_alignment"] = bool(int(params["use_anchor_alignment"]))
        model_config["anchor_weight"] = params["anchor_weight"]
        model_config["amp_weight"] = params["amp_weight"]
        model_config["proj_dim"] = params["proj_dim"]
        model_config["text_input_dim"] = params["text_input_dim"]

        # ===== 新增：给各模态 embedding 一个非 0 的大小 =====
        # 根据 data_config 动态决定当前数据集真正可用的模态
        modal_dims = resolve_modal_dims(dataset_name, data_config, params)
        model_config.update(modal_dims)


    print("Start init data")
    print(dataset_name, model_name, data_config, fold, batch_size)

    debug_print(text="init_dataset",fuc_name="main")

    # ========== 修改：正确获取 test_loader ==========
    test_loader = None
    test_window_loader = None

    if model_name not in ["dimkt"]:
        result = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size)
        if len(result) >= 2:
            train_loader, valid_loader = result[0], result[1]
        if len(result) >= 3:
            test_loader = result[2]
        if len(result) >= 4:
            test_window_loader = result[3]
    else:
        diff_level = params["difficult_levels"]
        result = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size, diff_level=diff_level)
        if len(result) >= 2:
            train_loader, valid_loader = result[0], result[1]
        if len(result) >= 3:
            test_loader = result[2]
        if len(result) >= 4:
            test_window_loader = result[3]
        # train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size, diff_level=diff_level)
    print("test_loader is None:", test_loader is None)

    params_str = "_".join([str(v) for k,v in params.items() if not k in ['other_config']])

    print(f"params: {params}, params_str: {params_str}")

    ckpt_path = os.path.join(save_dir, params_str)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    print(f"Start training model: {model_name}, embtype: {emb_type}, save_dir: {ckpt_path}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")

    if model_name in ["DyCalKT", "dycalkt"]:
        print("\n" + "=" * 60)
        print("[ABLATION CHECK]")
        print("ablation_mode =",
              model_config.get("ablation_mode"))
        print("use_cal_loss =",
              model_config.get("use_cal_loss"))
        print("use_con_loss =",
              model_config.get("use_con_loss"))
        print("=" * 60 + "\n")
        
    print(f"train_config: {train_config}")

    # ===== DEBUG: 检查 是否误开了 time / cluster =====
    # print("[DEBUG-model_config]")
    # print("num_diff =", model_config.get("num_diff", "NOT_FOUND"))
    # print("num_behavior =", model_config.get("num_behavior", "NOT_FOUND"))
    # print("num_time =", model_config.get("num_time", "NOT_FOUND"))
    # print("num_cluster =", model_config.get("num_cluster", "NOT_FOUND"))

    if model_config.get("num_time", 0) == 0:
        print("[CHECK] num_time == 0  -> time 模态已关闭")
    else:
        print("[CHECK] num_time != 0  -> time 模态被打开了")

    if model_config.get("num_cluster", 0) == 0:
        print("[CHECK] num_cluster == 0  -> cluster 模态已关闭")
    else:
        print("[CHECK] num_cluster != 0  -> cluster 模态被打开了")

    if model_name in ["dimkt"]:
        # del model_config['num_epochs']
        del model_config['weight_decay']

    save_config(train_config, model_config, data_config[dataset_name], params, ckpt_path)
    learning_rate = params["learning_rate"]
    
    remove_items = ['learning_rate', 'num_epochs', 'batch_size']
    # AKT 自己需要 l2，就不要提前删掉
    if model_name not in ["akt", "AKT", "ukt", "UKT"]:
        remove_items.append("l2")

    for remove_item in remove_items:
        model_config.pop(remove_item, None)
    
    if model_name in ["saint","saint++", "sakt", "atdkt", "simplekt", "DyCalKT", "dycalkt", "KeenKT", "bakt_time","folibikt", "ukt", "UKT"]:
        model_config["seq_len"] = seq_len


    debug_print(text = "init_model",fuc_name="main")
    print(f"model_name:{model_name}")

    model = init_model.init_model(model_name, model_config, data_config[dataset_name], emb_type)
    print(f"model is {model}")
    if model_name == "hawkes":
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, model.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optdict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        opt = torch.optim.Adam(optdict, lr=learning_rate, weight_decay=params['l2'])
    elif model_name == "iekt":
        opt = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6) 
        scheduler = StepLR(opt, step_size=10, gamma=0.1)  # 每10个epoch学习率乘以0.1
    elif model_name == "dtransformer":
        print(f"dtransformer weight_decay = 1e-5")
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif model_name == "dimkt":
        opt = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=params['weight_decay'])
    elif model_name == "DyCalKT":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        # opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    else:
        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            print(model.parameters())
            opt = Adam(model.parameters(), learning_rate)


    testauc, testacc = -1, -1
    window_testauc, window_testacc = -1, -1
    validauc, validacc = -1, -1
    best_epoch = -1
    save_model = True

    debug_print(text = "train model",fuc_name="main")


    if model_name == "rkt":
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = \
            train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model, data_config[dataset_name], fold)
    else:
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch, temps = train_model.train_model(
    model, train_loader, valid_loader, num_epochs, opt, ckpt_path,  save_model=save_model,test_loader=test_loader
)

    # if save_model:
    #     best_model = init_model.init_model(model_name, model_config, data_config[dataset_name], emb_type)
    #     net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
    #     best_model.load_state_dict(net)
    if save_model:
        ckpt_file = os.path.join(ckpt_path, emb_type + "_model.ckpt")
        print(f"[INFO] Loading checkpoint from {ckpt_file}")

        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(
                f"没有找到 checkpoint: {ckpt_file}\n"
                f"请检查 train_model.py 是否保存成了 {emb_type}_model.ckpt"
            )

        best_model = init_model.init_model(model_name, model_config, data_config[dataset_name], emb_type)
        net = torch.load(ckpt_file, map_location=device)
        best_model.load_state_dict(net)

    # print("\n================== 训练结束 ==================")
    # print("最终测试结果如下：")
    # print("="*70)
    # print(f"Best Epoch : {best_epoch}")
    # print(f"Best Valid AUC : {validauc:.4f}")
    # print(f"Final Test AUC : {testauc:.4f}")
    # print("="*70)


    file_path = f"./{dataset_name}_students_cov.txt"
    
    # 保存为 JSON 文件
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in temps:
            tensor_str = ' '.join(map(str, item.tolist()))  # 将 tensor 转换为字符串
            f.write(tensor_str + '\n')  # 每个 tensor 写入一行
    print(f"student cov for {dataset_name} has been saved to {file_path}")

