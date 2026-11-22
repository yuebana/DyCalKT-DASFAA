import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--model_name", type=str, default="DyCalKT")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    # ===== DyCalKT 消融实验 =====
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="full",
        choices=[
            "backbone",
            "static_evidence",
            "selection_only",
            "calibration_only",
            "full"
        ]
    )

    parser.add_argument(
        "--use_cal_loss",
        type=int,
        default=0,
        choices=[0, 1]
    )

    parser.add_argument(
        "--use_con_loss",
        type=int,
        default=0,
        choices=[0, 1]
    )
    # ===== DyCalKT 模型参数 =====
    parser.add_argument("--atten_type", type=str, default="dp")

    parser.add_argument("--use_dynamic_modal", type=int, default=1, choices=[0, 1])

    parser.add_argument("--use_anchor_alignment", type=int, default=1, choices=[0, 1])

    parser.add_argument("--anchor_weight", type=float, default=0.02)

    parser.add_argument("--amp_weight", type=float, default=0.3)

    parser.add_argument("--proj_dim", type=int, default=128)

    parser.add_argument("--text_input_dim", type=int, default=384)

    parser.add_argument("--final_fc_dim", type=int, default=512)
    parser.add_argument("--final_fc_dim2", type=int, default=256)

    # parser.add_argument("--use_wandb", type=int, default=0)
    # parser.add_argument("--add_uuid", type=int, default=0)

    args = parser.parse_args()

    params = vars(args)
    main(params)
