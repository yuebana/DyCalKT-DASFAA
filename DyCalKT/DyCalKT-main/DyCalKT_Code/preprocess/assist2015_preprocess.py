import pandas as pd
import numpy as np
from .utils import sta_infos, write_txt


KEYS = ["user_id", "sequence_id"]

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

def build_difficulty_map(df, skill_col="sequence_id", ans_col="correct", n_bins=10,
                         smooth_lambda=20, rare_thresh=5):
    """
    - 从“直接 accuracy 分桶”升级为“平滑 difficulty + rare 桶 + 分位数分桶”

    difficulty_id:
        0 = rare / unreliable sequence_id
        1~n_bins = regular difficulty bins
    """
    stat_src = df.copy()

    # 如果原始表带 fold，则只用非测试部分统计 difficulty
    if "fold" in df.columns:
        non_test_df = df[df["fold"] != -1].copy()
        if len(non_test_df) > 0:
            stat_src = non_test_df

    stat_df = stat_src.groupby(skill_col)[ans_col].agg(["sum", "count"]).reset_index()
    stat_df["acc"] = stat_df["sum"] / stat_df["count"]

    global_acc = stat_src[ans_col].mean()

    # 平滑正确率
    stat_df["smooth_acc"] = (
        stat_df["count"] * stat_df["acc"] + smooth_lambda * global_acc
    ) / (stat_df["count"] + smooth_lambda)

    # 默认 rare bucket = 0
    stat_df["difficulty_id"] = 0

    normal_mask = stat_df["count"] >= rare_thresh
    if normal_mask.any():
        diff_score = 1.0 - stat_df.loc[normal_mask, "smooth_acc"]
        diff_bins = safe_qcut_codes(diff_score, q=n_bins) + 1
        stat_df.loc[normal_mask, "difficulty_id"] = diff_bins.values.astype(int)


    return dict(zip(stat_df[skill_col], stat_df["difficulty_id"]))


def build_behavior_sequence(ans_list, window_size=3):
    """
    用“前 window_size 次作答正确率”构造代理 behavior 模态。
    注意：
        这里不是原始行为特征，而是当前文件条件下可用的代理行为模态。
    编码规则：
        0 -> 历史表现偏差（过去窗口正确率 < 1/3）
        1 -> 历史表现波动（1/3 <= 正确率 < 2/3）
        2 -> 历史表现较好（正确率 >= 2/3）
    第一个交互没有历史，默认记为 1（中性状态）。
    """
    behavior_seq = []

    for i in range(len(ans_list)):
        # 第一个位置没有历史信息，设为中性行为状态
        if i == 0:
            behavior_seq.append("1")
            continue

        start_idx = max(0, i - window_size)
        hist = ans_list[start_idx:i]
        hist_acc = sum(hist) / len(hist)

        if hist_acc < 1 / 3:
            behavior_seq.append("0")
        elif hist_acc < 2 / 3:
            behavior_seq.append("1")
        else:
            behavior_seq.append("2")

    return behavior_seq

def read_data_from_csv(read_file, write_file):
    stares = []

    df = pd.read_csv(read_file)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df["index"] = range(df.shape[0])

    df = df.dropna(subset=["user_id", "log_id", "sequence_id", "correct"])
    df = df[df['correct'].isin([0,1])]#filter responses 只保留二分类作答
    df['correct'] = df['correct'].astype(int)
    df['sequence_id'] = df['sequence_id'].astype(int)  # 统一 skill id 类型，便于后续映射 difficulty

    # =========================================================
    # 新增：为 assist2015 构造 difficulty 模态
    # difficulty 基于 sequence_id 的历史正确率分桶
    # =========================================================
    diff_map = build_difficulty_map(df, skill_col="sequence_id", ans_col="correct", n_bins=10)
    
    # 给每条交互映射 difficulty_id
    # 若个别 sequence_id 未命中映射，先用 5 作为中间难度兜底，避免 astype(int) 因 NaN 报错
    df["difficulty_id"] = df["sequence_id"].map(diff_map).fillna(5).astype(int)

    print("========[DEBUG-1] difficulty_id sample:")
    print(df[["user_id", "sequence_id", "correct", "difficulty_id"]].head(10))

    print("[DEBUG-1] difficulty unique values:", sorted(df["difficulty_id"].unique())[:20])


    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    ui_df = df.groupby('user_id', sort=False)

    user_inters = []
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]
        tmp_inter = tmp_inter.sort_values(by=["log_id", "index"]).reset_index(drop=True)
        seq_len = len(tmp_inter)

        # 原始 skill / response 序列
        seq_skills = tmp_inter["sequence_id"].astype(str).tolist()
        seq_ans = tmp_inter["correct"].astype(str).tolist()

        # 当前 assist2015_preprocess.py 里没有显式题目、时间、作答耗时字段
        # 因此这里仍先保留占位符，后续若原始数据有这些字段，再替换
        seq_problems = ["NA"]
        seq_start_time = ["NA"]
        seq_response_cost = ["NA"]

        # =========================================================
        # 新增：difficulty 序列：直接使用前面映射好的 difficulty_id
        # =========================================================
        seq_difficulties = tmp_inter["difficulty_id"].astype(str).tolist()

        # =========================================================
        # 新增：behavior 序列：用历史答题表现构造代理行为模态
        # =========================================================
        ans_int_list = tmp_inter["correct"].astype(int).tolist()
        seq_behaviors = build_behavior_sequence(ans_int_list, window_size=3)

        if len(user_inters) < 3:
            print(f"\n[DEBUG-2] user={user}, seq_len={seq_len}")
            print("=======[DEBUG-2] seq_skills[:10]      =", seq_skills[:10])
            print("[DEBUG-2] seq_ans[:10]         =", seq_ans[:10])
            print("[DEBUG-2] seq_difficulties[:10]=", seq_difficulties[:10])
            print("[DEBUG-2] seq_behaviors[:10]   =", seq_behaviors[:10])

        # 必须保证新增模态长度与 responses 一致，否则后续 dataloader 会出问题
        assert seq_len == len(seq_skills) == len(seq_ans) == len(seq_difficulties) == len(seq_behaviors)

        # user_inters 现在在原有 6 段基础上，额外附加两段：
        # 1) difficulty 序列
        # 2) behavior 序列
        user_inters.append(
            [
                [str(user), str(seq_len)],
                seq_problems,
                seq_skills,
                seq_ans,
                seq_start_time,
                seq_response_cost,
                seq_difficulties,
                seq_behaviors
            ]
        )
    print("\n[DEBUG-3] first user_inters sample:")
    for i, part in enumerate(user_inters[0]):
        print(f"part {i}: {part[:10]}")
    write_txt(write_file, user_inters)

    print("\n".join(stares))

    return

