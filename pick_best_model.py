"""
pick_best_model.py — results CSV에서 최고 test accuracy 모델명을 출력
사용: python pick_best_model.py <results_csv> [--candidates resnet34 resnet34d] [--attn none]
출력: 모델명 (예: resnet34d)
"""

import argparse
import sys
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('--candidates', nargs='+', default=['resnet34', 'resnet34d'])
    p.add_argument('--attn',       default='none')
    args = p.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"[pick_best_model] ERROR: {args.csv} not found", file=sys.stderr)
        sys.exit(1)

    filtered = df[df['model'].isin(args.candidates) & (df['attn'] == args.attn)]
    if filtered.empty:
        print(f"[pick_best_model] ERROR: no rows match", file=sys.stderr)
        sys.exit(1)

    best = filtered.loc[filtered['ensemble_test_acc'].idxmax()]
    print(best['model'])
    print(
        f"[pick_best_model] best: model={best['model']}, "
        f"test_acc={best['ensemble_test_acc']:.4f}",
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
