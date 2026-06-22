"""
進化的ヒューリスティック最適化スクリプト
v10の重みパラメータを進化アルゴリズムで最適化する。

戦略: (μ+λ) Evolution Strategy
  1. 重みベクトルの集団を作成
  2. 各個体をベースライン（v10）と対戦させて評価
  3. 上位を選抜 → 交叉+変異 → 次世代
  4. 繰り返し

使い方:
    python evolve.py [--generations 200] [--pop 30] [--games 50]
"""
import argparse
import random
import numpy as np
import json
import os

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, OptionType, SelectContext, AreaType

# ── デッキ ─────────────────────────────────────────────────────────────────

DECK_FIGHTING = (
    [673] * 2 + [674] * 2 + [675] * 1 + [676] * 3 + [677] * 4 + [678] * 4
  + [1102] * 4 + [1123] * 2 + [1141] * 4 + [1142] * 4 + [1152] * 4
  + [1159] * 1 + [1182] * 2 + [1192] * 4 + [1227] * 4 + [1252] * 2
  + [6] * 13
)

DECK_PSYCHIC = (
    [65] * 4 + [66] * 4 + [741] * 4 + [742] * 4 + [743] * 3
  + [1079] * 3 + [1081] * 3 + [1086] * 4 + [1097] * 1 + [1129] * 1
  + [1146] * 1 + [1152] * 4 + [1159] * 1 + [1182] * 3 + [1184] * 1
  + [1225] * 4 + [1231] * 4 + [1264] * 4
  + [5] * 3 + [19] * 4
)

DECK_GRASS = (
    [65] * 4 + [66] * 3 + [304] * 2 + [878] * 4 + [879] * 2
  + [1086] * 4 + [1097] * 3 + [1115] * 3 + [1122] * 4 + [1152] * 4
  + [1171] * 4 + [1182] * 2 + [1194] * 2 + [1210] * 2 + [1227] * 4 + [1255] * 4
  + [11] * 4 + [12] * 1 + [19] * 4
)

OPP_DECKS = [DECK_FIGHTING, DECK_PSYCHIC, DECK_GRASS]
MEGA_LUCARIO_ID = 678


# ── パラメータ化されたヒューリスティック ────────────────────────────────────

# 重みベクトルの定義（15個のパラメータ）
PARAM_NAMES = [
    'evolve',           # 0: 進化スコア
    'ability',          # 1: 特性スコア
    'attach_mega',      # 2: エネ付与(Mega Lucario)
    'attach_active',    # 3: エネ付与(アクティブ)
    'attach_bench',     # 4: エネ付与(ベンチ)
    'play',             # 5: カード使用
    'attack_base',      # 6: 攻撃基本スコア
    'retreat_score',    # 7: にげるスコア(条件満たすとき)
    'retreat_hp_mega',  # 8: にげる判定閾値(ベンチにMega時)
    'retreat_hp_low',   # 9: にげる判定閾値(低HP)
    'emergency_mult',   # 10: デッキ枯れ緊急ボーナス乗数
    'emergency_thresh', # 11: デッキ枯れ緊急閾値
    'mega_bonus',       # 12: Mega Lucarioの選択ボーナス
    'energy_weight',    # 13: エネルギー数の重み(poke_score)
    'hp_weight',        # 14: HP重み(poke_score)
]

# v10のデフォルト値
DEFAULT_PARAMS = np.array([
    9000,   # evolve
    8500,   # ability
    8400,   # attach_mega
    8100,   # attach_active
    8000,   # attach_bench
    7000,   # play
    6000,   # attack_base
    7500,   # retreat_score
    0.5,    # retreat_hp_mega
    0.3,    # retreat_hp_low
    400,    # emergency_mult
    10,     # emergency_thresh
    50000,  # mega_bonus
    1000,   # energy_weight
    1.0,    # hp_weight
], dtype=np.float64)

# パラメータの変異幅
MUTATION_SCALES = np.array([
    1000, 1000, 1000, 1000, 1000,  # scores
    1000, 1000, 1000,              # scores
    0.15, 0.15,                    # HP ratios
    100, 3,                        # emergency
    10000, 300, 0.3,               # poke_score weights
], dtype=np.float64)


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


def _poke_score(poke, params):
    if poke is None:
        return 0
    bonus = params[12] if poke.id == MEGA_LUCARIO_ID else 0
    return bonus + len(poke.energies) * params[13] + poke.hp * params[14]


def score_options(obs, params):
    """パラメータ化されたヒューリスティックスコア"""
    select = obs.select
    options = select.option
    state = obs.current
    our_idx = state.yourIndex if state else 0

    deck_remaining = state.players[our_idx].deckCount if state else 60
    emergency_bonus = max(0, (params[11] - deck_remaining) * params[10])

    scores = []
    for i, o in enumerate(options):
        if o.type == OptionType.NUMBER:
            score = o.number if o.number is not None else 0
        elif o.type == OptionType.YES:
            score = 1
        elif o.type == OptionType.NO:
            score = 0
        elif o.type == OptionType.EVOLVE:
            score = params[0]
        elif o.type == OptionType.ABILITY:
            score = params[1]
        elif o.type == OptionType.ATTACH:
            poke = _get_pokemon(state, o.inPlayArea, o.inPlayIndex, our_idx)
            if poke and poke.id == MEGA_LUCARIO_ID:
                score = params[2]
            elif o.inPlayArea == AreaType.ACTIVE:
                score = params[3]
            else:
                score = params[4]
        elif o.type == OptionType.PLAY:
            score = params[5]
        elif o.type == OptionType.ATTACK:
            score = params[6] + i + emergency_bonus
        elif o.type == OptionType.CARD:
            if select.context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE,
                                  SelectContext.SETUP_ACTIVE_POKEMON,
                                  SelectContext.ATTACH_FROM):
                pIdx = o.playerIndex if o.playerIndex is not None else our_idx
                poke = _get_pokemon(state, o.area, o.index, pIdx)
                score = _poke_score(poke, params)
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
                    if hp_ratio < params[8] and bench_has_mega:
                        score = params[7]
                    elif hp_ratio < params[9]:
                        score = params[7]
        elif o.type == OptionType.END:
            score = -1
        else:
            score = i
        scores.append(score)
    return scores


def play_action(obs, params, deck):
    """パラメータ化されたヒューリスティックで行動選択"""
    if obs.select is None:
        return list(deck)
    options = obs.select.option
    n = len(options)
    scores = score_options(obs, params)
    desc = sorted(range(n), key=lambda k: scores[k], reverse=True)
    return desc[:obs.select.maxCount]


# ── 対戦評価 ──────────────────────────────────────────────────────────────

def play_game(params_a, deck_a, params_b, deck_b):
    """2つのパラメータで対戦。params_aの勝利=1, 負け=0, 引分=0.5"""
    try:
        obs_dict, _ = battle_start(list(deck_a), list(deck_b))
    except Exception:
        return 0.5

    for _ in range(2000):
        try:
            obs = to_observation_class(obs_dict)
        except Exception:
            break

        if obs.current and getattr(obs.current, 'result', -1) >= 0:
            winner = obs.current.result
            try:
                battle_finish()
            except Exception:
                pass
            return 1.0 if winner == 0 else 0.0

        try:
            player = obs.current.yourIndex if obs.current else 0
            params = params_a if player == 0 else params_b
            deck   = deck_a   if player == 0 else deck_b
            action = play_action(obs, params, deck)
            obs_dict = battle_select(action)
        except Exception:
            break

    try:
        battle_finish()
    except Exception:
        pass
    return 0.5


def evaluate(params, n_games):
    """パラメータを複数デッキのベースラインと対戦させて勝率を返す"""
    wins = 0
    total = 0
    for _ in range(n_games):
        opp_deck = random.choice(OPP_DECKS)
        # 先攻後攻ランダム
        if random.random() < 0.5:
            result = play_game(params, DECK_FIGHTING, DEFAULT_PARAMS, opp_deck)
        else:
            result = 1.0 - play_game(DEFAULT_PARAMS, opp_deck, params, DECK_FIGHTING)
        wins += result
        total += 1
    return wins / max(total, 1)


# ── 進化アルゴリズム ──────────────────────────────────────────────────────

def mutate(parent, scale=1.0):
    """ガウス変異"""
    child = parent.copy()
    noise = np.random.randn(len(parent)) * MUTATION_SCALES * scale
    child += noise
    # HP閾値は0-1にクリップ
    child[8] = np.clip(child[8], 0.05, 0.95)
    child[9] = np.clip(child[9], 0.05, 0.95)
    # 緊急閾値は正の値
    child[10] = max(child[10], 10)
    child[11] = max(child[11], 1)
    return child


def crossover(parent_a, parent_b):
    """一様交叉"""
    child = parent_a.copy()
    mask = np.random.random(len(child)) < 0.5
    child[mask] = parent_b[mask]
    return child


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generations', type=int, default=200)
    parser.add_argument('--pop',         type=int, default=30)
    parser.add_argument('--games',       type=int, default=50)
    parser.add_argument('--elite',       type=int, default=6)
    parser.add_argument('--out',         type=str, default='model')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 初期集団: v10を中心にガウスノイズ
    population = [DEFAULT_PARAMS.copy()]
    for _ in range(args.pop - 1):
        population.append(mutate(DEFAULT_PARAMS, scale=0.5))

    best_fitness = 0.0
    best_params = DEFAULT_PARAMS.copy()

    print(f"進化開始: {args.generations}世代, 集団{args.pop}, 評価{args.games}試合/個体")
    print(f"ベースライン (v10) を相手に最適化")

    for gen in range(args.generations):
        # 評価
        fitness = []
        for i, params in enumerate(population):
            f = evaluate(params, args.games)
            fitness.append(f)

        # ソート
        ranked = sorted(range(len(population)), key=lambda k: fitness[k], reverse=True)
        top_fit = fitness[ranked[0]]
        avg_fit = np.mean(fitness)

        if top_fit > best_fitness:
            best_fitness = top_fit
            best_params = population[ranked[0]].copy()
            # 最良パラメータを保存
            result = {name: float(best_params[i]) for i, name in enumerate(PARAM_NAMES)}
            result['fitness'] = float(best_fitness)
            result['generation'] = gen
            with open(f'{args.out}/best_params.json', 'w') as f:
                json.dump(result, f, indent=2)
            np.save(f'{args.out}/best_params.npy', best_params)

        if (gen + 1) % 5 == 0 or gen == 0:
            print(f"Gen {gen+1:3d}: best={top_fit:.3f}, avg={avg_fit:.3f}, "
                  f"all-time-best={best_fitness:.3f}")

        # 選抜 + 次世代生成
        elite = [population[ranked[i]] for i in range(args.elite)]
        new_pop = list(elite)  # エリート保存

        while len(new_pop) < args.pop:
            if random.random() < 0.7:
                # 交叉 + 変異
                pa = random.choice(elite)
                pb = random.choice(elite)
                child = crossover(pa, pb)
                child = mutate(child, scale=0.3)
            else:
                # 変異のみ
                parent = random.choice(elite)
                child = mutate(parent, scale=0.5)
            new_pop.append(child)

        population = new_pop

    # 最終結果
    print(f"\n最適化完了!")
    print(f"最良適応度: {best_fitness:.3f} (v10基準50%からの改善)")
    print(f"\n最適パラメータ:")
    for i, name in enumerate(PARAM_NAMES):
        print(f"  {name:20s}: {best_params[i]:10.1f}  (v10: {DEFAULT_PARAMS[i]:10.1f})")
    print(f"\n保存: {args.out}/best_params.json, {args.out}/best_params.npy")


if __name__ == '__main__':
    main()
