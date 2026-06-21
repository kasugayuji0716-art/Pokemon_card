"""
自己対戦データ収集スクリプト
v10ヒューリスティックで両者を動かし、(盤面特徴量, 勝敗ラベル) を収集する。

使い方:
    python collect_data.py [--games 5000]
"""
import argparse
import random
import numpy as np

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, OptionType, SelectContext, AreaType

DECK = (
    [673] * 2 + [674] * 2 + [675] * 1 + [676] * 3 + [677] * 4 + [678] * 4
  + [1102] * 4 + [1123] * 2 + [1141] * 4 + [1142] * 4 + [1152] * 4
  + [1159] * 1 + [1182] * 2 + [1192] * 4 + [1227] * 4 + [1252] * 2
  + [6] * 13
)

MEGA_LUCARIO_ID = 678
FEATURE_DIM = 25


# ── 特徴量抽出 ─────────────────────────────────────────────────────────────

def extract_features(obs):
    """
    現在の盤面を25次元のfloat32ベクトルに変換する。
    obs.current.yourIndex を「現在選択する側」とみなす。
    """
    state = obs.current
    if state is None:
        return None

    our_idx = state.yourIndex
    opp_idx = 1 - our_idx
    us  = state.players[our_idx]
    opp = state.players[opp_idx]

    feats = []

    # --- 自分のアクティブ (3次元) ---
    a = (us.active[0] if us.active else None)
    feats += [
        a.hp / a.maxHp if (a and a.maxHp > 0) else 0.0,
        min(len(a.energies), 5) / 5.0 if a else 0.0,
        1.0 if (a and a.id == MEGA_LUCARIO_ID) else 0.0,
    ]

    # --- 自分のベンチ 3体 (9次元) ---
    bench = us.bench or []
    for i in range(3):
        b = bench[i] if i < len(bench) else None
        feats += [
            b.hp / b.maxHp if (b and b.maxHp > 0) else 0.0,
            min(len(b.energies), 5) / 5.0 if b else 0.0,
            1.0 if (b and b.id == MEGA_LUCARIO_ID) else 0.0,
        ]

    # --- 相手のアクティブ (2次元) ---
    oa = (opp.active[0] if opp.active else None)
    feats += [
        oa.hp / oa.maxHp if (oa and oa.maxHp > 0) else 0.0,
        min(len(oa.energies), 5) / 5.0 if oa else 0.0,
    ]

    # --- 相手のベンチ 3体 (3次元) ---
    opp_bench = opp.bench or []
    for i in range(3):
        ob = opp_bench[i] if i < len(opp_bench) else None
        feats += [ob.hp / ob.maxHp if (ob and ob.maxHp > 0) else 0.0]

    # --- グローバル (8次元) ---
    feats += [
        len(us.prize)  / 6.0,
        len(opp.prize) / 6.0,
        min(getattr(us,  'handCount', 0), 10) / 10.0,
        min(getattr(opp, 'handCount', 0), 10) / 10.0,
        getattr(us,  'deckCount', 0) / 60.0,
        getattr(opp, 'deckCount', 0) / 60.0,
        len(us.discard  or []) / 60.0,
        len(opp.discard or []) / 60.0,
    ]

    assert len(feats) == FEATURE_DIM, f"feature dim mismatch: {len(feats)}"
    return feats


# ── ヒューリスティック (v10) ───────────────────────────────────────────────

def _get_pokemon(state, area, index, player_idx):
    if state is None or index is None:
        return None
    try:
        ps = state.players[player_idx]
        if area == AreaType.BENCH:
            if ps.bench and 0 <= index < len(ps.bench):
                return ps.bench[index]
        elif area == AreaType.ACTIVE:
            if ps.active and ps.active[0] is not None:
                return ps.active[0]
    except (IndexError, TypeError, AttributeError):
        pass
    return None


def _poke_score(poke):
    if poke is None:
        return 0
    bonus = 50000 if poke.id == MEGA_LUCARIO_ID else 0
    return bonus + len(poke.energies) * 1000 + poke.hp


def _score_options(obs):
    select  = obs.select
    options = select.option
    state   = obs.current
    our_idx = state.yourIndex if state else 0

    deck_remaining  = state.players[our_idx].deckCount if state else 60
    emergency_bonus = max(0, (10 - deck_remaining) * 400)

    scores = []
    for i, o in enumerate(options):
        if o.type == OptionType.NUMBER:
            score = o.number if o.number is not None else 0
        elif o.type == OptionType.YES:
            score = 1
        elif o.type == OptionType.NO:
            score = 0
        elif o.type == OptionType.EVOLVE:
            score = 9000
        elif o.type == OptionType.ABILITY:
            score = 8500
        elif o.type == OptionType.ATTACH:
            poke = _get_pokemon(state, o.inPlayArea, o.inPlayIndex, our_idx)
            if poke and poke.id == MEGA_LUCARIO_ID:
                score = 8400
            elif o.inPlayArea == AreaType.ACTIVE:
                score = 8100
            else:
                score = 8000
        elif o.type == OptionType.PLAY:
            score = 7000
        elif o.type == OptionType.ATTACK:
            score = 6000 + i + emergency_bonus
        elif o.type == OptionType.CARD:
            if select.context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE,
                                  SelectContext.SETUP_ACTIVE_POKEMON,
                                  SelectContext.ATTACH_FROM):
                pIdx = o.playerIndex if o.playerIndex is not None else our_idx
                poke  = _get_pokemon(state, o.area, o.index, pIdx)
                score = _poke_score(poke)
                if pIdx != our_idx:
                    score = -score
            else:
                score = i
        elif o.type == OptionType.RETREAT:
            score = -2
            if state:
                ps = state.players[our_idx]
                if ps.active and ps.active[0] is not None:
                    a = ps.active[0]
                    hp_ratio = a.hp / a.maxHp if a.maxHp > 0 else 1.0
                    bench_has_mega = any(
                        b is not None and b.id == MEGA_LUCARIO_ID
                        for b in ps.bench
                    )
                    if hp_ratio < 0.5 and bench_has_mega:
                        score = 7500
                    elif hp_ratio < 0.3:
                        score = 7500
        elif o.type == OptionType.END:
            score = -1
        else:
            score = i
        scores.append(score)
    return scores


def heuristic_action(obs):
    if obs.select is None:
        return list(DECK)
    options = obs.select.option
    n       = len(options)
    scores  = _score_options(obs)
    desc    = sorted(range(n), key=lambda k: scores[k], reverse=True)
    return desc[:obs.select.maxCount]


# ── 1試合実行 ─────────────────────────────────────────────────────────────

def run_game():
    """
    1試合を v10 同士で回す。
    Returns:
        records: list of (features, player_idx)  各決定ステップの記録
        winner:  0 or 1  勝者プレイヤーインデックス (-1 = 未決)
    """
    try:
        obs_dict, _ = battle_start(list(DECK), list(DECK))
    except Exception as e:
        return [], -1

    records = []

    for _ in range(2000):  # 最大ステップ数 (無限ループ防止)
        try:
            obs = to_observation_class(obs_dict)
        except Exception:
            break

        # ゲーム終了確認
        if obs.current and getattr(obs.current, 'result', -1) >= 0:
            winner = obs.current.result
            try:
                battle_finish()
            except Exception:
                pass
            return records, winner

        # 特徴量を記録（select がある & state が取れているステップのみ）
        if obs.current and obs.select:
            feats = None
            try:
                feats = extract_features(obs)
            except Exception:
                pass
            if feats is not None:
                records.append((feats, obs.current.yourIndex))

        # 行動選択
        try:
            action   = heuristic_action(obs)
            obs_dict = battle_select(action)
        except Exception:
            break

    try:
        battle_finish()
    except Exception:
        pass
    return [], -1


# ── メイン ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=5000)
    parser.add_argument('--out',   type=str, default='model')
    args = parser.parse_args()

    all_X, all_y = [], []
    skip = 0

    print(f"{args.games} 試合を実行中...")
    for i in range(args.games):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{args.games} (サンプル数: {len(all_X)}, スキップ: {skip})")

        records, winner = run_game()
        if winner < 0:
            skip += 1
            continue

        for feats, player_idx in records:
            label = 1.0 if player_idx == winner else 0.0
            all_X.append(feats)
            all_y.append(label)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)

    import os
    os.makedirs(args.out, exist_ok=True)
    np.save(f'{args.out}/train_X.npy', X)
    np.save(f'{args.out}/train_y.npy', y)

    print(f"\n完了: {len(all_X)} サンプル保存 → {args.out}/train_X.npy")
    print(f"勝率バランス: {y.mean():.3f} (0.5 に近いほど良い)")
    print(f"スキップ: {skip} 試合")


if __name__ == '__main__':
    main()
