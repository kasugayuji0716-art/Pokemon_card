# pokeca-nn: ポケカAI 盤面評価ネットワーク

Kaggle「The Pokémon Company - PTCG AI Battle Challenge」用の
ニューラルネット評価関数の学習リポジトリ。

## セットアップ (Linux PC)

```bash
git clone https://github.com/<your-username>/pokeca-nn.git
cd pokeca-nn
pip install -r requirements.txt
```

## 使い方

### Step 1: 自己対戦データ収集

```bash
# デフォルト 5000 試合（約 10〜30 分）
python collect_data.py

# 試合数を増やす場合
python collect_data.py --games 20000
```

生成されるファイル:
- `model/train_X.npy` — 特徴量 (N × 25)
- `model/train_y.npy` — ラベル (N,)  1.0=勝ち / 0.0=負け

### Step 2: NN 学習

```bash
python train.py --epochs 100
```

生成されるファイル:
- `model/weights.npz` — 学習済み重み（numpy形式）

### Step 3: weights を Mac に転送

```bash
# このリポジトリに push する場合
git add model/weights.npz
git commit -m "add trained weights"
git push
```

---

## アーキテクチャ

```
入力 (25次元)
  自分のアクティブ: HP割合, エネルギー数, Mega Lucario?
  自分のベンチ×3: HP割合, エネルギー数, Mega Lucario?
  相手のアクティブ: HP割合, エネルギー数
  相手のベンチ×3: HP割合
  グローバル: 自サイド, 相手サイド, 手札数×2, デッキ残×2, 捨て×2

Linear(25→128) → ReLU → Linear(128→128) → ReLU → Linear(128→1) → Sigmoid

出力 (1次元): 勝率予測 [0.0, 1.0]
```

## ファイル構成

```
pokeca-nn/
├── cg/               ← Kaggle CGライブラリ (libcg.so含む)
├── collect_data.py   ← 自己対戦データ収集
├── train.py          ← NN学習
├── model/
│   ├── weights.npz   ← 学習済み重み (git管理)
│   └── .gitkeep
└── requirements.txt
```
