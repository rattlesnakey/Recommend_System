import argparse


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dataset_path", type=str, default='../data/train_small.zip')
    parser.add_argument("--ckpt_path", type=str, default='../checkpoints')
    parser.add_argument("--model", choices=['xDeepFM', 'DeepFM', 'ProXDeepFM'], default='ProXDeepFM')
    args = parser.parse_args()
    return args
    