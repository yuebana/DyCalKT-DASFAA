import os
import pandas as pd
import numpy as np
from .utils import sta_infos, write_txt, change2timestamp,format_list2str



def load_nips_data(primary_data_path,meta_data_dir,task_name):
    """The data downloaded from https://competitions.codalab.org/competitions/25449 
    The document can be downloaded from https://arxiv.org/abs/2007.12061.

    Args:
        primary_data_path (_type_): premary data path
        meta_data_dir (_type_): metadata dir
        task_name (_type_): task_1_2 or task_3_4

    Returns:
        dataframe: the merge df
    """
    print("Start load data")
    answer_metadata_path = os.path.join(meta_data_dir,f"answer_metadata_{task_name}.csv")
    question_metadata_path = os.path.join(meta_data_dir,f"question_metadata_{task_name}.csv")
    student_metadata_path = os.path.join(meta_data_dir,f"student_metadata_{task_name}.csv")
    subject_metadata_path = os.path.join(meta_data_dir,f"subject_metadata.csv")
    
    df_primary = pd.read_csv(primary_data_path)
    print(f"len df_primary is {len(df_primary)}")
    #add timestamp
    df_answer = pd.read_csv(answer_metadata_path)
    df_answer['answer_timestamp'] = df_answer['DateAnswered'].apply(change2timestamp)
    df_question = pd.read_csv(question_metadata_path)
    # df_student = pd.read_csv(student_metadata_path)
    df_subject = pd.read_csv(subject_metadata_path)
    
    #only keep level 3
    keep_subject_ids = set(df_subject[df_subject['Level']==3]['SubjectId'])
    df_question['SubjectId_level3'] = df_question['SubjectId'].apply(lambda x:set(eval(x))&keep_subject_ids)
    
    
    #merge data
    df_merge = df_primary.merge(df_answer[['AnswerId','answer_timestamp']],how='left')#merge answer time
    df_merge = df_merge.merge(df_question[["QuestionId","SubjectId_level3"]],how='left')#merge question subjects
    df_merge['SubjectId_level3_str'] = df_merge['SubjectId_level3'].apply(lambda x:"_".join([str(i) for i in x]))
    print(f"len df_merge is {len(df_merge)}")
    print("Finish load data")
    print(f"Num of student {df_merge['UserId'].unique().size}")
    print(f"Num of question {df_merge['QuestionId'].unique().size}")
    kcs =[]
    for item in df_merge['SubjectId_level3'].values:
        kcs.extend(item)
    print(f"Num of knowledge {len(set(kcs))}")
    return df_merge


def build_time_sequence_from_timestamps(ts_list):
    """
    用“相邻交互时间间隔”构造 time 模态。
    第一个交互没有前驱，记为 0 桶。
    后续位置使用 delta_t = t_i - t_{i-1}，再做离散化。

    分桶规则（单位：秒）：
        0: <= 1 min
        1: (1 min, 5 min]
        2: (5 min, 30 min]
        3: (30 min, 24 h]
        4: > 24 h
    """
    if len(ts_list) == 0:
        return []

    # 先保证是 int
    ts_list = [int(x) for x in ts_list]

    delta_list = [0]
    for i in range(1, len(ts_list)):
        delta_t = ts_list[i] - ts_list[i - 1]
        if delta_t < 0:
            delta_t = 0
        delta_list.append(delta_t)

    bins = [60, 300, 1800, 86400]
    time_ids = np.digitize(delta_list, bins=bins, right=True)

    return [str(int(x)) for x in time_ids]


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
    - 从“直接题目正确率分桶”升级为“平滑 difficulty + rare 桶 + 分位数分桶”

    difficulty_id:
        0 = rare / unreliable question
        1~5 = regular difficulty bins
    """
    tmp = df.copy()
    tmp["IsCorrect"] = tmp["IsCorrect"].astype(int)

    q_stats = tmp.groupby("QuestionId").agg(
        acc=("IsCorrect", "mean"),
        freq=("QuestionId", "count"),
    ).reset_index()

    global_acc = tmp["IsCorrect"].mean()

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

    question2diff = {
        int(qid): str(int(did))
        for qid, did in zip(q_stats["QuestionId"], q_stats["difficulty_id"])
    }
    return question2diff

def get_user_inters(df, question2diff):
    """convert df to user sequences"""
    user_inters = []
    for user, group in df.groupby("UserId", sort=False):
        group = group.sort_values(["answer_timestamp", "tmp_index"], ascending=True)

        seq_skills = group["SubjectId_level3_str"].tolist()
        seq_ans = group["IsCorrect"].astype(int).tolist()
        seq_start_time = group["answer_timestamp"].astype(int).tolist()
        seq_problems = group["QuestionId"].tolist()

        # 新增：difficulty 模态 = 题目历史正确率离散化
        seq_difficulties = [question2diff[int(q)] for q in seq_problems]

        # 当前方案：time 模态 = 相邻交互时间间隔离散化
        seq_times = build_time_sequence_from_timestamps(seq_start_time)

        # 这里仍保留原来的 usetimes 占位
        seq_response_cost = ["NA"]

        seq_len = len(group)

        assert seq_len == len(seq_skills) == len(seq_ans) == len(seq_start_time) == len(seq_problems) == len(seq_difficulties) == len(seq_times)

        user_inters.append(
            [
            [str(user), str(seq_len)],
            format_list2str(seq_problems),
            format_list2str(seq_skills),
            format_list2str(seq_ans),
            format_list2str(seq_start_time),
            format_list2str(seq_response_cost),
            format_list2str(seq_difficulties),
            format_list2str(seq_times),
            ]
        )
    return user_inters


KEYS = ["UserId", "SubjectId_level3_str", "QuestionId"]

def read_data_from_csv(primary_data_path,meta_data_dir,task_name,write_file):
    stares= []
    df = load_nips_data(primary_data_path,meta_data_dir,task_name)
    
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    df['tmp_index'] = range(len(df))
    df = df.dropna(subset=["UserId","answer_timestamp", "SubjectId_level3_str", "IsCorrect", "answer_timestamp","QuestionId"])

    df["IsCorrect"] = df["IsCorrect"].astype(int)

    # 新增：根据 QuestionId 构造 difficulty_id
    question2diff = build_question_difficulty_map(df)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    user_inters = get_user_inters(df, question2diff)
    write_txt(write_file, user_inters)
    
