"""
NN学習スクリプト v2
collect_data.py で生成したデータをもとに、盤面評価ネットワークを学習する。
出力: model/weights.npz (numpy形式, submission側でPyTorch不要)

改善点:
  - ラベル平滑化 (0.05/0.95)
  - Dropout (0.3/0.2/0.1)
  - AdamW + weight decay
  - ゲーム段階の重み付き損失
  - コサインアニーリングLRスケジューラ
  - 早期停止 (patience=20)

使い方:
    python train.py [--epochs 200] [--batch 512]
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

FEATURE_DIM = 96


def build_model(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     type=str,   default='model')
    parser.add_argument('--epochs',   type=int,   default=200)
    parser.add_argument('--batch',    type=int,   default=512)
    parser.add_argument('--lr',       type=float, default=1e-3)
    parser.add_argument('--patience', type=int,   default=20)
    args = parser.parse_args()

    # データ読み込み
    X_np = np.load(f'{args.data}/train_X.npy')
    y_np = np.load(f'{args.data}/train_y.npy')

    # 重みファイルがあれば使う（v2形式）
    w_path = f'{args.data}/train_w.npy'
    if os.path.exists(w_path):
        w_np = np.load(w_path)
        print("ゲーム段階重み: あり")
    else:
        w_np = np.ones_like(y_np)
        print("ゲーム段階重み: なし（均等重み）")

    print(f"データ: {len(X_np)} サンプル, {X_np.shape[1]} 次元")
    print(f"ラベル平均 (勝率): {y_np.mean():.3f}")

    # ラベル平滑化: 0/1 → 0.05/0.95
    y_np = np.clip(y_np, 0.05, 0.95)

    X = torch.FloatTensor(X_np)
    y = torch.FloatTensor(y_np)
    w = torch.FloatTensor(w_np)

    # 訓練/検証分割 (9:1)
    n_val = max(1, int(len(X) * 0.1))
    perm  = torch.randperm(len(X))
    X_val, y_val, w_val = X[perm[:n_val]], y[perm[:n_val]], w[perm[:n_val]]
    X_tr,  y_tr,  w_tr  = X[perm[n_val:]], y[perm[n_val:]], w[perm[n_val:]]

    dataset = TensorDataset(X_tr, y_tr, w_tr)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model     = build_model(X.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn   = nn.BCELoss(reduction='none')

    print(f"\n学習開始 ({args.epochs} エポック, patience={args.patience})...")
    print(f"モデル: {sum(p.numel() for p in model.parameters())} パラメータ")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb, wb in loader:
            pred = model(xb).squeeze()
            raw_loss = loss_fn(pred, yb)
            loss = (raw_loss * wb).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val).squeeze()
                val_raw  = loss_fn(val_pred, y_val)
                val_loss = (val_raw * w_val).mean().item()
                # 精度計算はラベル平滑化前の基準で
                val_acc  = ((val_pred > 0.5) == (y_val > 0.5)).float().mean().item()
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:3d}: train_loss={total_loss/len(loader):.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, lr={lr_now:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                weights = {k: v.numpy() for k, v in model.state_dict().items()}
                np.savez(f'{args.data}/weights.npz', **weights)
            else:
                patience_counter += 1
                if patience_counter >= args.patience // 10:
                    print(f"  (patience: {patience_counter * 10}/{args.patience})")

        # 早期停止（10エポックごとにチェックしているので patience÷10 で判定）
        if patience_counter >= args.patience // 10:
            print(f"\n早期停止 (patience={args.patience} エポック到達)")
            break

    print(f"\n保存完了: {args.data}/weights.npz (val_loss={best_val_loss:.4f})")
    print("このファイルを Mac に転送して submission に含めてください。")


if __name__ == '__main__':
    main()
