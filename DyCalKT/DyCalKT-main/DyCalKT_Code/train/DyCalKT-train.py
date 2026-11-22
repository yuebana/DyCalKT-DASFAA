import argparse
from wandb_train import main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--model_name", type=str, default="DyCalKT")
    parser.add_argument("--emb_type", type=str, default="stoc_qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--final_fc_dim", type=int, default=256)
    parser.add_argument("--final_fc_dim2", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--loss1", type=float, default=0.5)
    parser.add_argument("--loss2", type=float, default=0.5)
    parser.add_argument("--loss3", type=float, default=0.5)
    parser.add_argument("--start", type=int, default=50)


    parser.add_argument("--noise_level",    type=float, default=0.3)

    parser.add_argument("--atten_type", type= str, default="w2")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument('--use_dynamic_modal', type=int, default=1, choices=[0, 1], help='Use dynamic modality modeling')
    parser.add_argument('--use_anchor_alignment', type=int, default=1, choices=[0, 1], help='Use anchor alignment loss')
    parser.add_argument('--ablate_diff', type=int, default=0, help='1=关闭 difficulty, 0=使用 difficulty')
    parser.add_argument('--ablate_behavior', type=int, default=0, help='1=关闭 behavior, 0=使用 behavior')
    parser.add_argument('--ablate_cluster', type=int, default=0, help='1=关闭 cluster, 0=使用 cluster')
    parser.add_argument('--ablate_time', type=int, default=0, help='1=关闭 time, 0=使用 time')

    parser.add_argument('--dynamic_weight', type=float, default=0.08, help='Weight for dynamic loss')
    parser.add_argument('--anchor_weight', type=float, default=0.1, help='Weight for anchor alignment loss')
    parser.add_argument('--amp_weight', type=float, default=1.0, help='Weight for AMP loss')
    parser.add_argument('--threshold', type=float, default=0.08, help = 'selective dynamic threshold')
    parser.add_argument('--proj_dim', type=int, default=64, help='Projection dimension for alignment')
    parser.add_argument('--text_input_dim', type=int, default=384, help='Text feature dimension')

    # 先加上这些参数入口，后面再由 wandb_train.py 根据数据集补真实值
    parser.add_argument('--num_skill', type=int, default=0)
    parser.add_argument('--num_diff', type=int, default=0)
    parser.add_argument('--num_time', type=int, default=0)
    parser.add_argument('--num_cluster', type=int, default=0)
    parser.add_argument('--num_behavior', type=int, default=0)

    args = parser.parse_args()


    params = vars(args)
    main(params)
