#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
from .utils import sta_infos, write_txt, format_list2str, change2timestamp, replace_text

KEYS = ["Anon Student Id", "KC(SubSkills)", "Questions"]

def safe_qcut_codes(values, q):
    values = pd.Series(values)
    if len(values) == 0:
        return pd.Series(dtype=int)

    ranks = values.rank(method="first")
    try:
        codes = pd.qcut(ranks, q=q, labels=False, duplicates="drop")
        codes = pd.Series(codes).fillna(0).astype(int)
    except Exception:
        codes = pd.Series(np.zeros(len(values), dtype=int))
    return codes



def build_question_difficulty_map(df, smooth_lambda=20, n_bins=5, rare_thresh=5):
    """
    - 但从“直接正确率分桶”升级为“平滑 difficulty + rare 桶 + 分位数分桶”
    difficulty_id:
        0 = rare / unreliable question
        1~5 = regular difficulty bins
    """
    tmp = df.copy()
    tmp["Correct First Attempt"] = tmp["Correct First Attempt"].astype(int)

    q_stats = tmp.groupby("Questions").agg(
        acc=("Correct First Attempt", "mean"),
        freq=("Questions", "count"),
    ).reset_index()

    global_acc = tmp["Correct First Attempt"].mean()

    # 平滑正确率
    q_stats["smooth_acc"] = (
        q_stats["freq"] * q_stats["acc"] + smooth_lambda * global_acc
    ) / (q_stats["freq"] + smooth_lambda)

    q_stats["difficulty_id"] = 0   # rare bucket

    normal_mask = q_stats["freq"] >= rare_thresh
    if normal_mask.any():
        diff_score = 1.0 - q_stats.loc[normal_mask, "smooth_acc"]
        diff_bins = safe_qcut_codes(diff_score, q=n_bins) + 1
        q_stats.loc[normal_mask, "difficulty_id"] = diff_bins.values.astype(int)

    question2diff = {
        str(q): str(int(did))
        for q, did in zip(q_stats["Questions"], q_stats["difficulty_id"])
    }

    print("[ENHANCED-diff] num_questions_with_diff =", len(question2diff))
    print("[ENHANCED-diff] global_acc =", round(global_acc, 6))
    print("[ENHANCED-diff] unique difficulty ids =", sorted(set(question2diff.values())))

    return question2diff


def build_question_cluster_map(df, smooth_lambda=20, rare_thresh=5):
    """
    - 从“raw acc + freq + kc_num”升级为
      “smooth difficulty + log-frequency + kc_num”的粗粒度 cluster

    cluster_id:
        0 = rare / unreliable question
        1~12 = regular cluster
    """
    tmp = df.copy()
    tmp["Correct First Attempt"] = tmp["Correct First Attempt"].astype(int)
    tmp["kc_num"] = tmp["KC(SubSkills)"].apply(lambda x: len(str(x).split("~~")))

    q_stats = tmp.groupby("Questions").agg(
        acc=("Correct First Attempt", "mean"),
        freq=("Questions", "count"),
        kc_num=("kc_num", "mean"),
    ).reset_index()

    global_acc = tmp["Correct First Attempt"].mean()

    # 平滑正确率
    q_stats["smooth_acc"] = (
        q_stats["freq"] * q_stats["acc"] + smooth_lambda * global_acc
    ) / (q_stats["freq"] + smooth_lambda)

    # 默认 rare bucket = 0
    q_stats["cluster_id"] = 0

    normal_mask = q_stats["freq"] >= rare_thresh
    if normal_mask.any():
        qn = q_stats.loc[normal_mask].copy()

        # 1) 平滑 difficulty 分 3 桶
        qn["diff_score"] = 1.0 - qn["smooth_acc"]
        qn["diff_bin"] = pd.cut(
            qn["diff_score"],
            bins=[-1, 0.33, 0.66, 1.0],
            labels=[0, 1, 2]
        ).astype(int)

        # 2) log 频次分 2 桶
        qn["freq_bin"] = safe_qcut_codes(np.log1p(qn["freq"]), q=2)

        # 3) 知识点数分 2 桶
        def map_kc_bin(x):
            if x <= 1:
                return 0
            else:
                return 1

        qn["kc_bin"] = qn["kc_num"].apply(map_kc_bin)

        # regular cluster: 1~12
        qn["cluster_id"] = 1 + qn["diff_bin"] * 4 + qn["freq_bin"] * 2 + qn["kc_bin"]

        q_stats.loc[normal_mask, "cluster_id"] = qn["cluster_id"].values.astype(int)

    question2cluster = {
        str(q): str(int(cid))
        for q, cid in zip(q_stats["Questions"], q_stats["cluster_id"])
    }

    # print("[ENHANCED-cluster] min cluster id =", q_stats["cluster_id"].min())
    # print("[ENHANCED-cluster] max cluster id =", q_stats["cluster_id"].max())
    # print("[ENHANCED-cluster] num unique cluster ids =", q_stats["cluster_id"].nunique())
    # print("[ENHANCED-cluster] num_questions_with_cluster =", len(question2cluster))
    # print("[ENHANCED-cluster] unique cluster ids =", sorted(set(question2cluster.values())))

    return question2cluster


def read_data_from_csv(read_file, write_file):
    stares = []

    df = pd.read_table(read_file, low_memory=False)
    #concat problem name & step_name
    df["Problem Name"] = df["Problem Name"].apply(replace_text)
    df["Step Name"] = df["Step Name"].apply(replace_text)
    df["Questions"] = df.apply(lambda x:f"{x['Problem Name']}----{x['Step Name']}",axis=1)
    
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares, '~~')
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df["index"] = df.index
    df = df.dropna(subset=["Anon Student Id", "KC(SubSkills)", "Questions", "Correct First Attempt", "First Transaction Time"])
    #filter invalid record
    df = df[df["Correct First Attempt"].isin([0,1])]
    
    # 1. 先转成 datetime 类型
    df["First Transaction Time"] = pd.to_datetime(df["First Transaction Time"])
    # 2. 转成毫秒级时间戳字符串（和报错里的格式匹配）
    df["First Transaction Time"] = (df["First Transaction Time"].astype("int64") // 10**6).astype(str)
    # df.loc[:, "First Transaction Time"] = df.loc[:, "First Transaction Time"].apply(lambda t: change2timestamp(str(t)))
    df["KC(SubSkills)"] = df["KC(SubSkills)"].apply(replace_text)

    # df.loc[:, "First Transaction Time"] = df.loc[:, "First Transaction Time"].apply(lambda t: change2timestamp(t))
    # df["KC(SubSkills)"] = df["KC(SubSkills)"].apply(replace_text)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares, "~~")
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    question2cluster = build_question_cluster_map(df)
    question2diff = build_question_difficulty_map(df)

    user_inters = []
    for user, group in df.groupby("Anon Student Id", sort=False):
        group = group.sort_values(by=["First Transaction Time", "index"], ascending=True)
        group["First Transaction Time"] = group["First Transaction Time"].astype(str)

        seq_problems = group["Questions"].tolist()
        seq_ans = group["Correct First Attempt"].tolist()
        seq_start_time = group["First Transaction Time"].tolist()
        seq_skills = [x.replace("~~","_") for x in group["KC(SubSkills)"].tolist()]
        seq_clusters = [question2cluster[str(q)] for q in seq_problems]
        seq_difficulties = [question2diff[str(q)] for q in seq_problems]
        seq_len = len(seq_ans)
        seq_use_time = ["NA"]

        assert seq_len == len(seq_problems) == len(seq_skills) == len(seq_ans) == len(seq_start_time) == len(seq_clusters) == len(seq_difficulties)
        
        user_inters.append(
            [
                [user, str(seq_len)],
                seq_problems,
                seq_skills,
                format_list2str(seq_ans),
                seq_start_time,
                seq_use_time,
                format_list2str(seq_difficulties),
                format_list2str(seq_clusters),
            ]
        )
    write_txt(write_file, user_inters)

    print("\n".join(stares))

    return

