import pandas as pd
import numpy as np
from .utils import sta_infos, write_txt, change2timestamp, format_list2str

KEYS = ["User", "Problem"]


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


def build_problem_cluster_map(df, smooth_lambda=20, rare_thresh=5):
    """
    - 从“raw acc + freq + avg_attempts_per_user”升级为
      “smooth difficulty + log-frequency + avg_attempts_per_user”的粗粒度 cluster

    cluster_id:
        0 = rare / unreliable problem
        1~12 = regular cluster
    """
    tmp = df.copy()
    tmp["Result"] = tmp["Result"].astype(int)

    q_stats = tmp.groupby("Problem").agg(
        acc=("Result", "mean"),
        freq=("Problem", "count"),
        user_cnt=("User", "nunique"),
    ).reset_index()

    global_acc = tmp["Result"].mean()

    # 平滑正确率
    q_stats["smooth_acc"] = (
        q_stats["freq"] * q_stats["acc"] + smooth_lambda * global_acc
    ) / (q_stats["freq"] + smooth_lambda)

    q_stats["avg_attempts_per_user"] = q_stats["freq"] / q_stats["user_cnt"]

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

        # 3) 平均尝试次数分 2 桶
        qn["attempt_bin"] = safe_qcut_codes(qn["avg_attempts_per_user"], q=2)

        # regular cluster: 1~12
        qn["cluster_id"] = 1 + qn["diff_bin"] * 4 + qn["freq_bin"] * 2 + qn["attempt_bin"]

        q_stats.loc[normal_mask, "cluster_id"] = qn["cluster_id"].values.astype(int)

    problem2cluster = {
        str(p): str(int(cid))
        for p, cid in zip(q_stats["Problem"], q_stats["cluster_id"])
    }

    # print("[ENHANCED-cluster] min cluster id =", q_stats["cluster_id"].min())
    # print("[ENHANCED-cluster] max cluster id =", q_stats["cluster_id"].max())
    # print("[ENHANCED-cluster] num unique cluster ids =", q_stats["cluster_id"].nunique())
    # print("[ENHANCED-cluster] num_problems_with_cluster =", len(problem2cluster))
    # print("[ENHANCED-cluster] unique cluster ids sample =", sorted(set(problem2cluster.values()))[:20])

    return problem2cluster

def build_problem_difficulty_map(df, smooth_lambda=20, n_bins=5, rare_thresh=5):
    """
    - 从“直接 Problem 正确率分桶”升级为“平滑 difficulty + rare 桶 + 分位数分桶”

    difficulty_id:
        0 = rare / unreliable problem
        1~5 = regular difficulty bins
    """
    tmp = df.copy()
    tmp["Result"] = tmp["Result"].astype(int)

    q_stats = tmp.groupby("Problem").agg(
        acc=("Result", "mean"),
        freq=("Problem", "count"),
    ).reset_index()

    global_acc = tmp["Result"].mean()

    # 平滑正确率
    q_stats["smooth_acc"] = (
        q_stats["freq"] * q_stats["acc"] + smooth_lambda * global_acc
    ) / (q_stats["freq"] + smooth_lambda)

    # 默认 rare bucket = 0
    q_stats["difficulty_id"] = 0

    normal_mask = q_stats["freq"] >= rare_thresh
    if normal_mask.any():
        diff_score = 1.0 - q_stats.loc[normal_mask, "smooth_acc"]
        diff_bins = safe_qcut_codes(diff_score, q=n_bins) + 1
        q_stats.loc[normal_mask, "difficulty_id"] = diff_bins.values.astype(int)

    problem2diff = {
        str(p): str(int(did))
        for p, did in zip(q_stats["Problem"], q_stats["difficulty_id"])
    }

    # print("[ENHANCED-diff] num_problems_with_diff =", len(problem2diff))
    # print("[ENHANCED-diff] global_acc =", round(global_acc, 6))
    # print("[ENHANCED-diff] unique difficulty ids =", sorted(set(problem2diff.values())))

    return problem2diff
    
def read_data_from_csv(read_file, write_file):
    stares = []
    df = pd.read_csv(read_file)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    df["index"] = range(df.shape[0])    
    df = df.dropna(subset=["User", "Problem", "Result", "Submit Time"])
    df = df[df["Result"].isin(['Accepted', 'Wrong Answer', 'Compile Error', 'Time Limit Exceeded', 
            'Memory Limit Exceeded', 'Runtime Error', 'Output Limit Exceeded',
            'Presentation Error', 'System Error', 'Validator Error'])]
            # 'Waiting' 'Running & Judging' 'Compiling''])]
    df.loc[:, "Result"] = df.loc[:, "Result"].apply(lambda k: "1" if k == "Accepted" else "0")

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    problem2cluster = build_problem_cluster_map(df)
    problem2diff = build_problem_difficulty_map(df)
    
    data = []
    ui_df = df.groupby('User', sort=False)

    for ui in ui_df:
        uid, curdf = ui[0], ui[1].copy()

        curdf["Submit Time"] = curdf["Submit Time"].astype("object")
        curdf["Submit Time"] = curdf["Submit Time"].apply(lambda t: int(change2timestamp(t, False)))

        curdf = curdf.sort_values(by=["Submit Time", "index"])

        # 全部先转成标准 list
        seq_concepts = curdf["Problem"].astype(str).tolist()
        seq_responses = curdf["Result"].astype(str).tolist()
        seq_timestamps = curdf["Submit Time"].astype(str).tolist()

        # cluster 序列
        seq_clusters = [problem2cluster[str(p)] for p in seq_concepts]
        seq_difficulties = [problem2diff[str(p)] for p in seq_concepts]

        # poj 当前沿用原框架：questions 占位
        questions = ["NA"]
        usetimes = ["NA"]
        seq_len = len(seq_responses)
        uids = [str(uid), str(seq_len)]

        assert seq_len == len(seq_concepts) == len(seq_responses) == len(seq_timestamps) == len(seq_clusters) == len(seq_difficulties)

        data.append([
            uids,
            questions,
            format_list2str(seq_concepts),
            format_list2str(seq_responses),
            format_list2str(seq_timestamps),
            usetimes,
            format_list2str(seq_clusters),
            format_list2str(seq_difficulties),
        ])

        if len(data) % 1000 == 0:
            print(len(data))
    
    write_txt(write_file, data)

    print("\n".join(stares))

    return

