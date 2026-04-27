"""
pick_best_data.py — results CSV에서 최고 test accuracy 데이터 설정을 출력
사용: python pick_best_data.py <results_csv> [--model resnet34] [--attn none]

출력: --use_excluded --use_added 형태의 플래그 문자열 (없으면 빈 문자열)
"""

import argparse
import sys
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv', help='results CSV 경로')
    p.add_argument('--model', default='resnet34')
    p.add_argument('--attn',  default='none')
    args = p.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"[pick_best_data] ERROR: {args.csv} not found", file=sys.stderr)
        sys.exit(1)

    filtered = df[(df['model'] == args.model) & (df['attn'] == args.attn)]
    if filtered.empty:
        print(f"[pick_best_data] ERROR: no rows match model={args.model} attn={args.attn}",
              file=sys.stderr)
        sys.exit(1)

    best = filtered.loc[filtered['ensemble_test_acc'].idxmax()]

    flags = []
    if best['use_excluded']:
        flags.append('--use_excluded')
    if best['use_added']:
        flags.append('--use_added')
    if best['tuning']:
        flags.append('--tuning')

    print(' '.join(flags))

    # stderr에 선택 근거 출력 (sh 로그에서 확인용)
    print(
        f"[pick_best_data] best: use_excluded={bool(best['use_excluded'])}, "
        f"use_added={bool(best['use_added'])}, "
        f"test_acc={best['ensemble_test_acc']:.4f}",
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
