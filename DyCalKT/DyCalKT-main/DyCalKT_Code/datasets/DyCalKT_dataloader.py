import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import numpy as np
import math

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
else:
    from torch import FloatTensor, LongTensor

class DyCalKTDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).
    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """
    def __init__(self, file_path, input_type, folds,p=0.5, qtest=False):
        super(DyCalKTDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        self.p = p
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + folds_str + "_qtest.pkl"
        else:
            processed_data = file_path + folds_str + ".pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                
                self.dori, self.dqtest = self.__load_data__(sequence_path, folds)
                save_data = [self.dori, self.dqtest]
            else:
                self.dori = self.__load_data__(sequence_path, folds)
                save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori = pd.read_pickle(processed_data)
                for key in self.dori:
                    self.dori[key] = self.dori[key]#[:100]
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):
        """return the dataset length
        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get
        Returns:
            (tuple): tuple containing:
            
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation

            - **rshft_aug(torch.tensor)
        """

        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if not key == "uid":
                if key in ["masks", "smasks"]:
                    continue
                if len(self.dori[key]) == 0:
                    dcur[key] = self.dori[key]
                    dcur["shft_"+key] = self.dori[key]
                    continue
                # print(f"key: {key}, len: {len(self.dori[key])}")
                seqs = self.dori[key][index][:-1] * mseqs
                shft_seqs = self.dori[key][index][1:] * mseqs
                dcur[key] = seqs
                dcur["shft_"+key] = shft_seqs


        rshft_aug = dcur["shft_rseqs"].clone()
        r_aug     = dcur["rseqs"].clone()
        rshft_aug_mask = rshft_aug[mseqs]
        r_aug_mask = r_aug[mseqs]

        if rshft_aug_mask[-1] == 1:
            if r_aug_mask[0] == 1:
               r_aug[0] = 1 - r_aug[0]
            for i in range(0, len(rshft_aug_mask)):
                if rshft_aug[0] == 1:
                    rshft_aug[i] = 1 - rshft_aug[i]
        elif rshft_aug_mask[-1] == 0:
            if r_aug_mask[0] == 0:
               r_aug[0] = 1 - r_aug[0]
            for i in range(0, len(rshft_aug_mask)):
                if rshft_aug[0] == 0:
                    rshft_aug[i] = 1 - rshft_aug[i]
                


        dcur["shft_r_aug"] = rshft_aug
        dcur["r_aug"] = r_aug


        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]

                # 新增：记录当前样本可用的动态模态
        dcur["has_skill"] = torch.tensor(int(torch.any(dcur["skillseqs"] != -1))) if "skillseqs" in dcur else torch.tensor(0)
        dcur["has_diff"] = torch.tensor(int(torch.any(dcur["diffseqs"] != -1))) if "diffseqs" in dcur else torch.tensor(0)
        dcur["has_time"] = torch.tensor(int(torch.any(dcur["timeseqs"] != -1))) if "timeseqs" in dcur else torch.tensor(0)
        dcur["has_cluster"] = torch.tensor(int(torch.any(dcur["clusterseqs"] != -1))) if "clusterseqs" in dcur else torch.tensor(0)
        dcur["has_behavior"] = torch.tensor(int(torch.any(dcur["behaviorseqs"] != -1))) if "behaviorseqs" in dcur else torch.tensor(0)

        
        if not self.qtest:  
            return dcur
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dqtest



    


    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]): 
            pad_val (int, optional): pad value. Defaults to -1.
        Returns: 
            (tuple): tuple containing
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        dori = {
            "qseqs": [],         # question id 序列（主干里原本就用）
            "cseqs": [],         # concept / skill id 序列（主干里原本就用）
            "rseqs": [],         # response 序列

            # 原始时间相关字段
            "tseqs": [],         # timestamps
            "utseqs": [],        # usetimes

            # ===== 新增：为动态模态统一预留字段 =====
            "skillseqs": [],     # skill / concept 模态
            "diffseqs": [],      # difficulty 模态
            "timeseqs": [],      # 时间模态（统一给模型使用）
            "clusterseqs": [],   # cluster 模态
            "behaviorseqs": [],  # behavior 模态（可为多维连续特征）

            "smasks": [],
            "uid": []
        }
        # dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": [],"uid":[]}

        # seq_qids, seq_cids, seq_rights, seq_mask = [], [], [], []
        df = pd.read_csv(sequence_path)#[0:1000]
        df = df[df["fold"].isin(folds)]
        interaction_num = 0
        # seq_qidxs, seq_rests = [], []
        dqtest = {"qidxs": [], "rests":[], "orirow":[]}
        
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                dori["cseqs"].append([int(_) for _ in row["concepts"].split(",")])
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
                
           
            # 1) skill 模态
            # 优先使用 skill / skills 列；如果没有，可退化为 concepts
            if "skill" in row.index:
                dori["skillseqs"].append([int(_) for _ in str(row["skill"]).split(",")])
            elif "skills" in row.index:
                dori["skillseqs"].append([int(_) for _ in str(row["skills"]).split(",")])
            elif "concepts" in row.index:
                # 对很多 KT 数据集来说 concepts 本身就可视为 skill 模态
                dori["skillseqs"].append([int(_) for _ in str(row["concepts"]).split(",")])
            else:
                dori["skillseqs"].append([])

            # 2) difficulty 模态
            if "difficulty" in row.index:
                dori["diffseqs"].append([int(_) for _ in str(row["difficulty"]).split(",")])
            elif "difficulties" in row.index:
                dori["diffseqs"].append([int(_) for _ in str(row["difficulties"]).split(",")])
            else:
                dori["diffseqs"].append([])

            # 3) time 模态
            # 统一存到 timeseqs，优先 timestamps，其次 usetimes
            if "timestamps" in row.index:
                dori["timeseqs"].append([int(_) for _ in str(row["timestamps"]).split(",")])
            elif "usetimes" in row.index:
                dori["timeseqs"].append([int(_) for _ in str(row["usetimes"]).split(",")])
            else:
                dori["timeseqs"].append([])

            # 4) cluster 模态
            if "cluster" in row.index:
                dori["clusterseqs"].append([int(_) for _ in str(row["cluster"]).split(",")])
            elif "clusters" in row.index:
                dori["clusterseqs"].append([int(_) for _ in str(row["clusters"]).split(",")])
            else:
                dori["clusterseqs"].append([])

            # 5) behavior 模态
            if "behavior" in row.index:
                dori["behaviorseqs"].append([int(_) for _ in str(row["behavior"]).split(",")])
            elif "behaviors" in row.index:
                dori["behaviorseqs"].append([int(_) for _ in str(row["behaviors"]).split(",")])
            else:
                dori["behaviorseqs"].append([])
                 
            if "uid" in row:
                dori["uid"].append(row["uid"])    

                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])

        # =====================================================
        # 新增：将缺失模态补成与 qseqs 等长的 pad 序列
        # 避免空列表无法转 Tensor
        # =====================================================
        seq_ref = dori["qseqs"] if len(dori["qseqs"]) > 0 else dori["cseqs"]

        for idx in range(len(seq_ref)):
            seq_len = len(seq_ref[idx])

            # 对这些动态模态，如果为空，则补成 pad_val
            for modal_key in ["skillseqs", "diffseqs", "timeseqs", "clusterseqs", "behaviorseqs"]:
                if len(dori[modal_key][idx]) == 0:
                    dori[modal_key][idx] = [pad_val] * seq_len

                    
        for key in dori:
            if key not in ["rseqs"]:#in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["cseqs"][:,:-1] != pad_val) * (dori["cseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs
        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            
            return dori, dqtest
        
        return dori
    
#  a = 0 b = 1

def change_random_A_to_B(lst, shft_lst, a,p):
    for i in range(1, len(lst)):
        #num = np.random.randint(0,len(lst))
        prob = np.random.rand()
        if prob < p:
            if lst[i] == a:
                lst[i] = 1 - lst[i]
                shft_lst[i-1] = 1 - shft_lst[i-1]
                break
        #if lst[num] == a:
            #lst[num] = b
            #shft_lst[num+1] = b
            #break
        
    return lst, shft_lst



