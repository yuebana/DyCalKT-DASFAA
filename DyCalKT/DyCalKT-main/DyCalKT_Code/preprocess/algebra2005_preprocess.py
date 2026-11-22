#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
from .utils import sta_infos, write_txt, format_list2str, change2timestamp, replace_text

KEYS = ["Anon Student Id", "KC(Default)", "Questions"]

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
    - 从“直接正确率分桶”升级为“平滑 difficulty + rare 桶 + 分位数分桶”

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

    return question2diff



def map_interaction_difficulty_from_concepts(kc_str, concept2diff, default_val=0):
    """
    一个交互如果对应多个 concept，就把这些 concept 的 difficulty 取平均，再四舍五入。
    """
    concepts = [c for c in str(kc_str).split("~~") if c != ""]
    if len(concepts) == 0:
        return str(default_val)

    vals = [concept2diff.get(c, default_val) for c in concepts]
    diff_id = int(round(sum(vals) / len(vals)))
    return str(diff_id)

def read_data_from_csv(read_file, write_file):
    stares= []

    df = pd.read_table(read_file, encoding = "utf-8", low_memory=False)
    df["Problem Name"] = df["Problem Name"].apply(replace_text)
    df["Step Name"] = df["Step Name"].apply(replace_text)
    df["Questions"] = df.apply(lambda x:f"{x['Problem Name']}----{x['Step Name']}",axis=1)

    #判断有没有multi-skill
    print("[DEBUG-cols] columns=", df.columns.tolist())
    if "KC(SubSkills)" in df.columns:
        subskill_has_multi = df["KC(SubSkills)"].dropna().astype(str).str.contains("~~", regex=False).any()
        print("[DEBUG-cols] KC(SubSkills) has '~~' ? ->", subskill_has_multi)

    if "KC(Default)" in df.columns:
        default_has_multi = df["KC(Default)"].dropna().astype(str).str.contains("~~", regex=False).any()
        print("[DEBUG-cols] KC(Default) has '~~' ? ->", default_has_multi)
 
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares, '~~')
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    

    df["index"] = range(df.shape[0])
    df = df.dropna(subset=["Anon Student Id", "Questions", "KC(Default)", "First Transaction Time", "Correct First Attempt"])
    df = df[df["Correct First Attempt"].isin([0,1])]
    df = df[["index", "Anon Student Id", "Questions", "KC(Default)", "First Transaction Time", "Correct First Attempt"]]
    df["KC(Default)"] = df["KC(Default)"].apply(replace_text)

    #确认清洗后~~还在
    print("[DEBUG-after-clean] sample KC(Default) =",
          df["KC(Default)"].dropna().astype(str).head(10).tolist())

    print("[DEBUG-after-clean] KC(Default) has '~~' ? ->",
          df["KC(Default)"].dropna().astype(str).str.contains("~~", regex=False).any())

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares, "~~")
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    question2diff = build_question_difficulty_map(df)


    data = []
    ui_df = df.groupby('Anon Student Id', sort=False)

    for ui in ui_df:
        u, curdf = ui[0], ui[1].copy()

        # 先把时间列转成 object，再转成整数时间戳，避免字符串列赋值报错
        curdf["First Transaction Time"] = curdf["First Transaction Time"].astype("object")
        curdf["First Transaction Time"] = curdf["First Transaction Time"].apply(lambda t: int(change2timestamp(t)))

        # 按时间和原始索引排序
        curdf = curdf.sort_values(by=["First Transaction Time", "index"])

        # 写出前再转回字符串
        curdf["First Transaction Time"] = curdf["First Transaction Time"].astype(str)

        seq_problems = curdf["Questions"].astype(str).tolist()
        # seq_skills = curdf["KC(Default)"].astype(str).tolist()   # 保留 ~~ ，不要替换
        seq_skills = [x.replace("~~", "_") for x in curdf["KC(Default)"].astype(str).tolist()]   # 把~~转成_
        if len(data) < 2:
            print("[DEBUG-seq] raw KC(Default) sample =", curdf["KC(Default)"].astype(str).tolist()[:3])
            print("[DEBUG-seq] seq_skills sample =", seq_skills[:3])
        seq_ans = curdf["Correct First Attempt"].astype(str).tolist()
        seq_start_time = curdf["First Transaction Time"].astype(str).tolist()
        seq_difficulties = [question2diff[str(q)] for q in seq_problems]


        seq_len = len(seq_ans)
        seq_use_time = ["NA"]

        assert seq_len == len(seq_problems) == len(seq_skills) == len(seq_ans) == len(seq_start_time) == len(seq_difficulties)

        data.append([
            [str(u), str(seq_len)],
            format_list2str(seq_problems),
            format_list2str(seq_skills),
            format_list2str(seq_ans),
            format_list2str(seq_start_time),
            seq_use_time,
            format_list2str(seq_difficulties),
        ])
    write_txt(write_file, data)

    print("\n".join(stares))

    return

