"""
対戦ログスクリプト
v16(進化パラメータ) vs v10(デフォルト) を対戦させ、
各ステップの行動を詳細に記録する。

使い方:
    python battle_log.py [--games 5]
"""
import argparse
import random
import json
import os

from cg.game import battle_start, battle_select, battle_finish
from cg.api import (to_observation_class, OptionType, SelectContext, AreaType,
                    all_card_data, all_attack)

# ── カードDB ──────────────────────────────────────────────────────────────

CARD_DB = {}
ATTACK_DB = {}
try:
    for c in all_card_data():
        CARD_DB[c.cardId] = c
    for a in all_attack():
        ATTACK_DB[a.attackId] = a
except Exception:
    pass

def card_name(card_id):
    cd = CARD_DB.get(card_id)
    return f"{cd.name}({card_id})" if cd else f"ID{card_id}"

def poke_str(poke):
    if poke is None:
        return "[empty]"
    cd = CARD_DB.get(poke.id)
    name = cd.name if cd else f"ID{poke.id}"
    ex_tag = ""
    if cd:
        if cd.megaEx: ex_tag = " [MegaEX]"
        elif cd.ex: ex_tag = " [EX]"
    energies = len(poke.energies) if poke.energies else 0
    return f"{name}{ex_tag} HP:{poke.hp}/{poke.maxHp} E:{energies}"

def option_str(opt):
    parts = [opt.type.name]
    if opt.type == OptionType.CARD:
        cd = CARD_DB.get(opt.cardId) if opt.cardId else None
        if cd:
            parts.append(cd.name)
    return " ".join(parts)

# ── デッキ ────────────────────────────────────────────────────────────────

DECK_FIGHTING = (
    [673]*2 + [674]*2 + [675]*1 + [676]*3 + [677]*4 + [678]*4
  + [1102]*4 + [1123]*2 + [1141]*4 + [1142]*4 + [1152]*4
  + [1159]*1 + [1182]*2 + [1192]*4 + [1227]*4 + [1252]*2
  + [6]*13
)

DECK_PSYCHIC = (
    [65]*4 + [66]*4 + [741]*4 + [742]*4 + [743]*3
  + [1079]*3 + [1081]*3 + [1086]*4 + [1097]*1 + [1129]*1
  + [1146]*1 + [1152]*4 + [1159]*1 + [1182]*3 + [1184]*1
  + [1225]*4 + [1231]*4 + [1264]*4
  + [5]*3 + [19]*4
)

DECK_GRASS = (
    [65]*4 + [66]*3 + [304]*2 + [878]*4 + [879]*2
  + [1086]*4 + [1097]*3 + [1115]*3 + [1122]*4 + [1152]*4
  + [1171]*4 + [1182]*2 + [1194]*2 + [1210]*2 + [1227]*4 + [1255]*4
  + [11]*4 + [12]*1 + [19]*4
)

MEGA_LUCARIO_ID = 678

# ── パラメータ定義 ────────────────────────────────────────────────────────

V10_PARAMS = {
    'evolve': 9000, 'ability': 8500,
    'attach_mega': 8400, 'attach_active': 8100, 'attach_bench': 8000,
    'play': 7000, 'attack_base': 6000,
    'retreat_score': 7500, 'retreat_hp_mega': 0.5, 'retreat_hp_low': 0.3,
    'emergency_mult': 400, 'emergency_thresh': 10,
    'mega_bonus': 50000, 'energy_weight': 1000, 'hp_weight': 1.0,
}

V16_PARAMS = None  # best_params.jsonから読み込み

def load_v16_params():
    path = 'model/best_params.json'
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if k not in ('fitness', 'generation')}
    return None

# ── パラメータ化ヒューリスティック ────────────────────────────────────────

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
    bonus = params['mega_bonus'] if poke.id == MEGA_LUCARIO_ID else 0
    return bonus + len(poke.energies) * params['energy_weight'] + poke.hp * params['hp_weight']

def score_options(obs, params):
    select = obs.select
    options = select.option
    state = obs.current
    our_idx = state.yourIndex if state else 0
    deck_remaining = state.players[our_idx].deckCount if state else 60
    emergency_bonus = max(0, (params['emergency_thresh'] - deck_remaining) * params['emergency_mult'])

    scores = []
    for i, o in enumerate(options):
        if o.type == OptionType.NUMBER:
            score = o.number if o.number is not None else 0
        elif o.type == OptionType.YES:
            score = 1
        elif o.type == OptionType.NO:
            score = 0
        elif o.type == OptionType.EVOLVE:
            score = params['evolve']
        elif o.type == OptionType.ABILITY:
            score = params['ability']
        elif o.type == OptionType.ATTACH:
            poke = _get_pokemon(state, o.inPlayArea, o.inPlayIndex, our_idx)
            if poke and poke.id == MEGA_LUCARIO_ID:
                score = params['attach_mega']
            elif o.inPlayArea == AreaType.ACTIVE:
                score = params['attach_active']
            else:
                score = params['attach_bench']
        elif o.type == OptionType.PLAY:
            score = params['play']
        elif o.type == OptionType.ATTACK:
            score = params['attack_base'] + i + emergency_bonus
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
                    if hp_ratio < params['retreat_hp_mega'] and bench_has_mega:
                        score = params['retreat_score']
                    elif hp_ratio < params['retreat_hp_low']:
                        score = params['retreat_score']
        elif o.type == OptionType.END:
            score = -1
        else:
            score = i
        scores.append(score)
    return scores

def play_action(obs, params, deck):
    if obs.select is None:
        return list(deck)
    options = obs.select.option
    n = len(options)
    scores = score_options(obs, params)
    desc = sorted(range(n), key=lambda k: scores[k], reverse=True)
    return desc[:obs.select.maxCount]


# ── 対戦ログ ──────────────────────────────────────────────────────────────

def run_logged_game(params_0, name_0, deck_0, params_1, name_1, deck_1, verbose=True):
    """詳細ログ付き対戦"""
    try:
        obs_dict, _ = battle_start(list(deck_0), list(deck_1))
    except Exception as e:
        print(f"  battle_start failed: {e}")
        return -1

    step = 0
    last_turn = -1

    for _ in range(2000):
        try:
            obs = to_observation_class(obs_dict)
        except Exception:
            break

        state = obs.current
        if state and getattr(state, 'result', -1) >= 0:
            winner = state.result
            winner_name = name_0 if winner == 0 else name_1
            if verbose:
                # 最終盤面
                for pi in range(2):
                    ps = state.players[pi]
                    pname = name_0 if pi == 0 else name_1
                    a = ps.active[0] if ps.active else None
                    bench_count = len([b for b in (ps.bench or []) if b])
                    prize_left = len(ps.prize)
                    print(f"  [{pname}] Prize:{prize_left} Active:{poke_str(a)} Bench:{bench_count}")
                print(f"  ★ 勝者: {winner_name}")
            try:
                battle_finish()
            except Exception:
                pass
            return winner

        if state and obs.select:
            player_idx = state.yourIndex
            player_name = name_0 if player_idx == 0 else name_1
            params = params_0 if player_idx == 0 else params_1
            deck = deck_0 if player_idx == 0 else deck_1

            turn = getattr(state, 'turn', -1)
            options = obs.select.option
            n = len(options)

            if verbose and turn != last_turn:
                last_turn = turn
                # ターン開始時の盤面表示
                print(f"\n--- Turn {turn} ({player_name}'s action) ---")
                for pi in range(2):
                    ps = state.players[pi]
                    pname = name_0 if pi == 0 else name_1
                    a = ps.active[0] if ps.active else None
                    bench_list = [poke_str(b) for b in (ps.bench or []) if b]
                    prize_left = len(ps.prize)
                    deck_left = ps.deckCount
                    print(f"  [{pname}] Prize:{prize_left} Deck:{deck_left} "
                          f"Active:{poke_str(a)}")
                    if bench_list:
                        print(f"    Bench: {', '.join(bench_list)}")

            # 行動選択
            scores = score_options(obs, params)
            desc = sorted(range(n), key=lambda k: scores[k], reverse=True)
            action = desc[:obs.select.maxCount]
            chosen_idx = action[0] if action else 0
            chosen_opt = options[chosen_idx] if chosen_idx < n else None

            if verbose and chosen_opt:
                opt_type = chosen_opt.type.name
                score_val = scores[chosen_idx]

                # 選択肢の概要
                opt_types = [o.type.name for o in options]
                opt_scores = [f"{o.type.name}({scores[i]:.0f})" for i, o in enumerate(options)]

                # 重要な選択（ATTACK, RETREAT, EVOLVE等）のみ表示
                if chosen_opt.type in (OptionType.ATTACK, OptionType.RETREAT,
                                        OptionType.EVOLVE, OptionType.ABILITY,
                                        OptionType.END):
                    print(f"  → {player_name}: {opt_type} (score:{score_val:.0f})")
                    if len(opt_scores) <= 8:
                        print(f"    候補: {', '.join(opt_scores)}")

        try:
            params_cur = params_0 if (state and state.yourIndex == 0) else params_1
            deck_cur = deck_0 if (state and state.yourIndex == 0) else deck_1
            action = play_action(obs, params_cur, deck_cur)
            obs_dict = battle_select(action)
        except Exception:
            break
        step += 1

    try:
        battle_finish()
    except Exception:
        pass
    return -1


# ── メイン ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=5)
    parser.add_argument('--opp-deck', type=str, default='fighting',
                        choices=['fighting', 'psychic', 'grass'])
    args = parser.parse_args()

    global V16_PARAMS
    V16_PARAMS = load_v16_params()
    if V16_PARAMS is None:
        print("model/best_params.json が見つかりません")
        return

    opp_decks = {
        'fighting': DECK_FIGHTING,
        'psychic': DECK_PSYCHIC,
        'grass': DECK_GRASS,
    }
    opp_deck = opp_decks[args.opp_deck]

    print(f"=== v16(進化パラメータ) vs v10(デフォルト) ===")
    print(f"デッキ: v16=Fighting, v10={args.opp_deck}")
    print(f"試合数: {args.games}\n")

    v16_wins = 0
    v10_wins = 0

    for game_i in range(args.games):
        print(f"\n{'='*60}")
        print(f"Game {game_i+1}/{args.games}")
        print(f"{'='*60}")

        # 先攻後攻をランダムに
        if random.random() < 0.5:
            winner = run_logged_game(
                V16_PARAMS, "v16", DECK_FIGHTING,
                V10_PARAMS, "v10", opp_deck
            )
            if winner == 0: v16_wins += 1
            elif winner == 1: v10_wins += 1
        else:
            winner = run_logged_game(
                V10_PARAMS, "v10", opp_deck,
                V16_PARAMS, "v16", DECK_FIGHTING
            )
            if winner == 0: v10_wins += 1
            elif winner == 1: v16_wins += 1

    print(f"\n{'='*60}")
    print(f"結果: v16 {v16_wins}勝 - v10 {v10_wins}勝 "
          f"({args.games - v16_wins - v10_wins}引分)")
    print(f"v16勝率: {v16_wins / max(v16_wins + v10_wins, 1):.1%}")


if __name__ == '__main__':
    main()
