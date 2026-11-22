import os
import sys
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
import numpy as np
from .evaluate_model import evaluate
from train.config import que_type_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_epoch_summary(
    epoch,
    train_loss,
    valid_auc,
    valid_acc,
    best_auc,
    best_acc,
    best_epoch,
    model_name,
    emb_type,
    save_dir,
    test_auc=-1,
    test_acc=-1,
    window_test_auc=-1,
    window_test_acc=-1
):
    print("=" * 80)
    print(f"Epoch: {epoch}")
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Valid AUC: {valid_auc:.4f}, Valid ACC: {valid_acc:.4f}")
    print(f"Best AUC: {best_auc:.4f}, Best ACC: {best_acc:.4f} (Epoch {best_epoch})")
    print(f"Model: {model_name}, Embedding Type: {emb_type}")
    print(f"Save Directory: {save_dir}")
    # print(f"Test AUC: {test_auc}, Test ACC: {test_acc}, Window Test AUC: {window_test_auc}, Window Test ACC: {window_test_acc}")
    print("=" * 80)

def cal_loss(model, ys, r, rshft, sm, preloss=None):
    model_name = model.model_name
    preloss = preloss if preloss is not None else []

    def _base_bce(y_pred, target, sm):
        y_valid = y_pred[sm]
        t_valid = target[sm]
        return binary_cross_entropy(y_valid.double(), t_valid.double())

    if model_name in ["simplekt", "sakt", "saint"]:
        return _base_bce(ys[0], rshft, sm)

    elif model_name in ["dkt"]:
        loss = _base_bce(ys[0], rshft, sm)
        if len(preloss) > 0:
            loss = loss + sum(preloss)
        return loss

    elif model_name in ["DyCalKT", "dycalkt"]:
        if not ys:
            raise ValueError("DyCalKT模型未返回有效输出，请检查forward方法")

        bce_loss = _base_bce(ys[0], rshft, sm)
        loss = bce_loss
        if len(preloss) > 0:
            loss = loss + sum(preloss)
        return loss

    elif model_name in ["akt", "AKT"]:
        if len(ys) < 1 or len(preloss) < 1:
            raise ValueError("AKT模型输出结构不完整")
        return _base_bce(ys[0], rshft, sm) + preloss[0]
    elif model_name in ["ukt", "UKT"]:
        if not ys:
            raise ValueError("UKT模型未返回有效输出，请检查forward方法")

        bce_loss = _base_bce(ys[0], rshft, sm)
        aux_loss = 0.0
        if model.use_CL and len(ys) > 1 and isinstance(ys[1], torch.Tensor):
            aux_loss = ys[1].mean()
        return bce_loss + model.cl_weight * aux_loss

    else:
        raise ValueError(f"不支持的模型: {model_name}")

def model_forward(model, data):
    model_name = model.model_name
    dcur = {k: v.to(device) for k, v in data.items()}

    q, c, r = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"]
    qshft, cshft, rshft, sm = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"], dcur["smasks"]

    diff = dcur["diffseqs"] if "diffseqs" in dcur else None
    time = dcur["timeseqs"] if "timeseqs" in dcur else None
    cluster = dcur["clusterseqs"] if "clusterseqs" in dcur else None
    behavior = dcur["behaviorseqs"] if "behaviorseqs" in dcur else None

    ys, preloss = [], []

    if model_name in ["simplekt"]:
        outputs = model(dcur, train=True)
        ys = [outputs[0][:, 1:]]

    # elif model_name in ["DyCalKT", "dycalkt"]:
    #     outputs = model(dcur, train=True)
    #     ys = [outputs[0][:, 1:]]

    elif model_name in ["DyCalKT", "dycalkt"]:
        outputs = model(dcur, train=True)

        # 主预测
        ys = [outputs[0][:, 1:]]

        # DyCalKT 内部辅助损失：
        # L_cal + L_con
        if len(outputs) > 1 and isinstance(outputs[1], torch.Tensor):
            preloss.append(outputs[1])

    elif model_name in ["dkt", "DKT"]:
        y_full, anchor_loss = model(
            c.long(),
            r.long(),
            diff=diff.long() if diff is not None else None,
            time=time.long() if time is not None else None,
            cluster=cluster.long() if cluster is not None else None,
            behavior=behavior.long() if behavior is not None else None,
            return_aux=True
        )

        # DKT 输出是对每个 concept 的概率，取 shift 后目标 concept 的预测值
        y = (y_full * nn.functional.one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)

        if model.use_anchor_alignment:
            preloss.append(model.anchor_weight * anchor_loss)

    elif model_name in ["akt", "AKT"]:
        # 先拼完整序列，长度与 AKT 预测逻辑对齐
        cq = torch.cat((q[:, 0:1], qshft), dim=1)
        cc = torch.cat((c[:, 0:1], cshft), dim=1)
        cr = torch.cat((r[:, 0:1], rshft), dim=1)

        diff_full = None
        if diff is not None and "shft_diffseqs" in dcur:
            diff_full = torch.cat((diff[:, 0:1], dcur["shft_diffseqs"].to(device)), dim=1)
        elif diff is not None:
            diff_full = diff

        cluster_full = None
        if cluster is not None and "shft_clusterseqs" in dcur:
            cluster_full = torch.cat((cluster[:, 0:1], dcur["shft_clusterseqs"].to(device)), dim=1)
        elif cluster is not None:
            cluster_full = cluster

        time_full = None
        if time is not None and "shft_timeseqs" in dcur:
            time_full = torch.cat((time[:, 0:1], dcur["shft_timeseqs"].to(device)), dim=1)
        elif time is not None:
            time_full = time

        behavior_full = None
        if behavior is not None and "shft_behaviorseqs" in dcur:
            behavior_full = torch.cat((behavior[:, 0:1], dcur["shft_behaviorseqs"].to(device)), dim=1)
        elif behavior is not None:
            behavior_full = behavior

        y, reg_loss = model(
            q_data=cc.long(),                               # concept / skill full seq
            target=cr.long(),                               # response full seq
            pid_data=cq.long() if getattr(model, "n_pid", 0) > 0 else None,  # question full seq
            diff_data=diff_full.long() if diff_full is not None else None,
            cluster_data=cluster_full.long() if cluster_full is not None else None,
            time_data=time_full.long() if time_full is not None else None,
            behavior_data=behavior_full.long() if behavior_full is not None else None,
        )

        ys.append(y[:, 1:])   # 现在长度会和 rshft / sm 对齐
        preloss.append(reg_loss)
    elif model_name in ["ukt", "UKT"]:
        outputs = model(dcur, train=True)
        if model.use_CL:
            if len(outputs) < 5:
                raise ValueError("UKT对比学习模式需要返回至少5个值")
            y_main, cl_loss, _, _, temp = outputs[:5]
            ys = [y_main[:, 1:], cl_loss]
        else:
            y_main = outputs[0]
            ys = [y_main[:, 1:]]

    loss = cal_loss(model, ys, r, rshft, sm, preloss)

    if model_name in ["ukt", "UKT"] and getattr(model, 'use_CL', False):
        return loss, temp.item()
    return loss

def train_model(model, train_loader, valid_loader, num_epochs, optimizer,
                ckpt_path, test_loader=None, save_model=True,window_test_loader=None, fold=None, 
                patience=10, log_interval=100):

    best_metrics = {
        "valid_auc": -1,
        "valid_acc": -1,
        "test_auc": -1,
        "test_acc": -1,
        "window_test_auc": -1,
        "window_test_acc": -1,
        "best_epoch": 0,
        "best_temps": []
    }
    best_epoch = 0
    wait = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = []
        temps = []

        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            if model.model_name in ["ukt", "UKT"] and getattr(model, 'use_CL', False):
                loss, temp = model_forward(model, data)
                temps.append(temp)
            else:
                loss = model_forward(model, data)

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}",
                      flush=True)

        model.eval()
        with torch.no_grad():
            valid_auc, valid_acc = evaluate(model, valid_loader, model.model_name)

        if valid_auc > best_metrics["valid_auc"] + 1e-4:
            best_metrics.update({
                "valid_auc": valid_auc,
                "valid_acc": valid_acc,
                "best_epoch": epoch,
                "best_temps": temps.copy()
            })
            wait = 0

            if save_model:
                # 保存成 wandb_train.py 后面要读取的名字，比如 qid_model.ckpt
                # 使用固定文件名，每次覆盖旧的最佳模型
                ckpt_name = f"{model.emb_type}_model.ckpt"
                ckpt_full_path = os.path.join(ckpt_path, ckpt_name)
                # 如果文件已存在，先删除（确保覆盖）
                if os.path.exists(ckpt_full_path):
                    os.remove(ckpt_full_path)

                torch.save(model.state_dict(), ckpt_full_path)
                print(f"[SAVE] 保存最佳模型到 {ckpt_full_path} epoch{epoch} (AUC={valid_auc:.4f})", flush=True)

            if test_loader:
                test_auc, test_acc = evaluate(model, test_loader, model.model_name)
                best_metrics["test_auc"] = test_auc
                best_metrics["test_acc"] = test_acc
            if window_test_loader is not None:
                window_test_auc, window_test_acc = evaluate(model, window_test_loader, model.model_name)
                best_metrics["window_test_auc"] = window_test_auc
                best_metrics["window_test_acc"] = window_test_acc
        else:
            wait += 1
            if wait >= patience:
                print(f"[STOP] 早停触发：连续{patience}轮未提升", flush=True)
                break

        log_epoch_summary(
            epoch=epoch,
            train_loss=np.mean(epoch_loss),
            valid_auc=valid_auc,
            valid_acc=valid_acc,
            best_auc=best_metrics["valid_auc"],
            best_acc=best_metrics["valid_acc"],
            best_epoch=best_metrics["best_epoch"],
            model_name=model.model_name,
            emb_type=model.emb_type,
            save_dir=ckpt_path,
            test_auc=best_metrics.get("test_auc", -1),
            test_acc = best_metrics.get("test_acc", -1) #修改后
            # test_acc=-1
        )
    final_test_auc, final_test_acc = evaluate(model, test_loader, model.model_name)
    
    best_metrics["test_auc"] = final_test_auc
    best_metrics["test_acc"] = final_test_acc

    # ========== 修复：返回正确的8个值 ==========
    testauc = best_metrics.get("test_auc", -1)
    testacc = best_metrics.get("test_acc", -1)
    window_testauc = best_metrics.get("window_test_auc", -1)  # 如果不存在就返回-1
    window_testacc = best_metrics.get("window_test_acc", -1)
    # window_testauc = -1  # 暂时没有计算窗口测试
    # window_testacc = -1
    validauc = best_metrics.get("valid_auc", -1)
    validacc = best_metrics.get("valid_acc", -1)
    best_epoch = best_metrics.get("best_epoch", -1)
    temps = best_metrics.get("best_temps", [])

    print("AUC:", max(validauc, final_test_auc))
    print("ACC:", max(validacc, final_test_acc))
    
    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch, temps

    # return best_metrics
