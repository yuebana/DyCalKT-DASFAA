# _*_ coding:utf-8 _*_

import pandas as pd
import numpy as np
from .utils import sta_infos, write_txt, format_list2str

KEYS = ["user_id", "skill_id", "problem_id"]


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
    - 从“直接 problem 正确率分桶”升级为“平滑 difficulty + rare 桶 + 分位数分桶”

    difficulty_id:
        0 = rare / unreliable problem
        1~5 = regular difficulty bins
    """
    tmp = df.copy()
    tmp["correct"] = tmp["correct"].astype(int)

    q_stats = tmp.groupby("problem_id").agg(
        acc=("correct", "mean"),
        freq=("problem_id", "count"),
    ).reset_index()

    global_acc = tmp["correct"].mean()

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
        str(pid): str(int(did))
        for pid, did in zip(q_stats["problem_id"], q_stats["difficulty_id"])
    }

    return problem2diff

def read_data_from_csv(read_file, write_file):
    stares = []

    df = pd.read_csv(read_file, encoding = 'utf-8', dtype=str, encoding_errors='ignore')
    # df = pd.read_csv(read_file, encoding = 'gbk', dtype=str)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    df['tmp_index'] = range(len(df))
    _df = df.dropna(subset=["user_id","problem_id", "skill_id", "correct", "order_id"])

    _df["correct"] = _df["correct"].astype(int)
    problem2diff = build_question_difficulty_map(_df)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(_df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    ui_df = _df.groupby('user_id', sort=False)

    user_inters = []
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]
        tmp_inter = tmp_inter.sort_values(by=['order_id','tmp_index'])
        seq_len = len(tmp_inter)
        seq_problems = tmp_inter['problem_id'].tolist()
        seq_skills = tmp_inter['skill_id'].tolist()
        seq_ans = tmp_inter['correct'].tolist()
        seq_difficulties = [problem2diff[str(pid)] for pid in seq_problems]
        seq_start_time = ['NA']
        seq_response_cost = ['NA']

        assert seq_len == len(seq_problems) == len(seq_skills) == len(seq_ans) == len(seq_difficulties)

        user_inters.append(
            [
                [str(user), str(seq_len)],
                format_list2str(seq_problems),
                format_list2str(seq_skills),
                format_list2str(seq_ans),
                seq_start_time,
                seq_response_cost,
                format_list2str(seq_difficulties),
            ]
        )

    write_txt(write_file, user_inters)

    print("\n".join(stares))

    return

