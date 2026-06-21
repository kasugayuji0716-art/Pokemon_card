"""
NN学習スクリプト
collect_data.py で生成したデータをもとに、盤面評価ネットワークを学習する。
出力: model/weights.npz (numpy形式, submission側でPyTorch不要)

使い方:
    python train.py [--epochs 100] [--batch 512]
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

FEATURE_DIM = 25


def build_model(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   type=str, default='model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch',  type=int, default=512)
    parser.add_argument('--lr',     type=float, default=1e-3)
    args = parser.parse_args()

    # データ読み込み
    X_np = np.load(f'{args.data}/train_X.npy')
    y_np = np.load(f'{args.data}/train_y.npy')
    print(f"データ: {len(X_np)} サンプル, {X_np.shape[1]} 次元")
    print(f"ラベル平均 (勝率): {y_np.mean():.3f}")

    X = torch.FloatTensor(X_np)
    y = torch.FloatTensor(y_np)

    # 訓練/検証分割 (9:1)
    n_val = max(1, int(len(X) * 0.1))
    perm  = torch.randperm(len(X))
    X_val, y_val   = X[perm[:n_val]], y[perm[:n_val]]
    X_tr,  y_tr    = X[perm[n_val:]], y[perm[n_val:]]

    dataset = TensorDataset(X_tr, y_tr)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model     = build_model(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn   = nn.BCELoss()

    print(f"\n学習開始 ({args.epochs} エポック)...")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val).squeeze()
                val_loss = loss_fn(val_pred, y_val).item()
                val_acc  = ((val_pred > 0.5) == (y_val > 0.5)).float().mean().item()
            print(f"Epoch {epoch+1:3d}: train_loss={total_loss/len(loader):.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # ベストモデルを保存
                weights = {k: v.numpy() for k, v in model.state_dict().items()}
                np.savez(f'{args.data}/weights.npz', **weights)

    print(f"\n保存完了: {args.data}/weights.npz (val_loss={best_val_loss:.4f})")
    print("このファイルを Mac に転送して submission に含めてください。")


if __name__ == '__main__':
    main()
