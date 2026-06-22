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
    [741]*4 + [742]*4 + [743]*4  # Abra→Kadabra→Alakazam
  + [305]*4 + [66]*4             # Dunsparce→Dudunsparce
  + [1079]*4 + [1081]*3 + [1086]*4 + [1097]*2 + [1129]*2
  + [1152]*4 + [1155]*1 + [1156]*2 + [1182]*2 + [1184]*1
  + [1225]*3 + [1231]*4
  + [5]*4 + [19]*4
)

DECK_GRASS = (
    [65]*4 + [66]*3 + [304]*2 + [878]*4 + [879]*2
  + [1086]*4 + [1097]*3 + [1115]*3 + [1122]*4 + [1152]*4
  + [1171]*4 + [1182]*2 + [1194]*2 + [1210]*2 + [1227]*4 + [1255]*4
  + [11]*4 + [12]*1 + [19]*4
)

MEGA_LUCARIO_ID = 678
ALAKAZAM_ID = 743

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
                print(f"\n{'━'*50}")
                print(f"  🏆 勝者: {winner_name}")
                for pi in range(2):
                    ps = state.players[pi]
                    pname = name_0 if pi == 0 else name_1
                    a = ps.active[0] if ps.active else None
                    bench_count = len([b for b in (ps.bench or []) if b])
                    prize_left = len(ps.prize)
                    deck_left = ps.deckCount
                    print(f"  [{pname}] サイド残:{prize_left} デッキ残:{deck_left} "
                          f"バトル場:{poke_str(a)} ベンチ:{bench_count}体")
                print(f"{'━'*50}")
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
                print(f"\n{'─'*50}")
                print(f"ターン {turn}  ({player_name}の番)")
                print(f"{'─'*50}")
                for pi in range(2):
                    ps = state.players[pi]
                    pname = name_0 if pi == 0 else name_1
                    a = ps.active[0] if ps.active else None
                    bench_list = [poke_str(b) for b in (ps.bench or []) if b]
                    prize_left = len(ps.prize)
                    deck_left = ps.deckCount
                    hand_count = getattr(ps, 'handCount', '?')
                    marker = "👉" if pi == player_idx else "  "
                    print(f"{marker}[{pname}] サイド:{prize_left} デッキ:{deck_left} 手札:{hand_count}")
                    print(f"    バトル場: {poke_str(a)}")
                    if bench_list:
                        for bi, bs in enumerate(bench_list):
                            print(f"    ベンチ{bi+1}: {bs}")

            # 行動選択
            scores = score_options(obs, params)
            desc = sorted(range(n), key=lambda k: scores[k], reverse=True)
            action = desc[:obs.select.maxCount]
            chosen_idx = action[0] if action else 0
            chosen_opt = options[chosen_idx] if chosen_idx < n else None

            if verbose and chosen_opt:
                try:
                    # カード名ヘルパー
                    def _opt_card_name(opt):
                        cid = getattr(opt, 'cardId', None)
                        if cid and cid in CARD_DB:
                            return CARD_DB[cid].name
                        return None

                    def _effect_card_name():
                        """SelectData.effect / contextCard からカード名を取得"""
                        sel = obs.select
                        for attr in ('effect', 'contextCard'):
                            card = getattr(sel, attr, None)
                            if card and hasattr(card, 'id') and card.id in CARD_DB:
                                return CARD_DB[card.id].name
                        return None

                    def _atk_name(opt):
                        aid = getattr(opt, 'attackId', None)
                        if aid and aid in ATTACK_DB:
                            return ATTACK_DB[aid].name
                        return None

                    action_desc = ""
                    if chosen_opt.type == OptionType.EVOLVE:
                        name = _opt_card_name(chosen_opt) or _effect_card_name() or "?"
                        action_desc = f"⬆ 進化: {name}"
                    elif chosen_opt.type == OptionType.PLAY:
                        name = _opt_card_name(chosen_opt) or _effect_card_name() or "?"
                        action_desc = f"🃏 使用: {name}"
                    elif chosen_opt.type == OptionType.ATTACH:
                        target = "アクティブ" if getattr(chosen_opt, 'inPlayArea', None) == AreaType.ACTIVE else "ベンチ"
                        poke = _get_pokemon(state, chosen_opt.inPlayArea,
                                           getattr(chosen_opt, 'inPlayIndex', None), player_idx)
                        pname = CARD_DB[poke.id].name if (poke and poke.id in CARD_DB) else "?"
                        action_desc = f"⚡ エネ付与 → {pname}({target})"
                    elif chosen_opt.type == OptionType.ATTACK:
                        aname = _atk_name(chosen_opt) or "?"
                        action_desc = f"⚔ 攻撃: {aname}"
                    elif chosen_opt.type == OptionType.RETREAT:
                        action_desc = f"🏃 にげる！"
                    elif chosen_opt.type == OptionType.ABILITY:
                        name = _effect_card_name() or "?"
                        action_desc = f"✨ 特性: {name}"
                    elif chosen_opt.type == OptionType.END:
                        action_desc = f"⏹ ターン終了"
                    elif chosen_opt.type == OptionType.CARD:
                        name = _opt_card_name(chosen_opt) or "?"
                        ctx = getattr(obs.select, 'context', None)
                        ctx_name = getattr(ctx, 'name', '') if ctx else ''
                        if name != "?":
                            action_desc = f"  📎 選択: {name}"
                        # カード名不明の選択は非表示（ノイズ削減）
                    elif chosen_opt.type in (OptionType.YES, OptionType.NO, OptionType.NUMBER):
                        pass  # 非表示
                    else:
                        action_desc = ""

                    if action_desc:
                        print(f"  {player_name}: {action_desc}")
                except Exception:
                    pass

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
    parser.add_argument('--deck-a', type=str, default='psychic',
                        choices=['fighting', 'psychic', 'grass'])
    parser.add_argument('--deck-b', type=str, default='fighting',
                        choices=['fighting', 'psychic', 'grass'])
    args = parser.parse_args()

    deck_map = {
        'fighting': (DECK_FIGHTING, MEGA_LUCARIO_ID),
        'psychic':  (DECK_PSYCHIC,  ALAKAZAM_ID),
        'grass':    (DECK_GRASS,    None),
    }
    deck_a, attacker_a = deck_map[args.deck_a]
    deck_b, attacker_b = deck_map[args.deck_b]

    # メインアタッカーIDに応じたパラメータ
    params_a = dict(V10_PARAMS)
    params_b = dict(V10_PARAMS)

    print(f"=== {args.deck_a} vs {args.deck_b} (両方v10ヒューリスティック) ===")
    print(f"試合数: {args.games}\n")

    a_wins = 0
    b_wins = 0

    for game_i in range(args.games):
        print(f"\n{'='*60}")
        print(f"Game {game_i+1}/{args.games}")
        print(f"{'='*60}")

        if random.random() < 0.5:
            winner = run_logged_game(
                params_a, args.deck_a, deck_a,
                params_b, args.deck_b, deck_b
            )
            if winner == 0: a_wins += 1
            elif winner == 1: b_wins += 1
        else:
            winner = run_logged_game(
                params_b, args.deck_b, deck_b,
                params_a, args.deck_a, deck_a
            )
            if winner == 0: b_wins += 1
            elif winner == 1: a_wins += 1

    print(f"\n{'='*60}")
    print(f"結果: {args.deck_a} {a_wins}勝 - {args.deck_b} {b_wins}勝 "
          f"({args.games - a_wins - b_wins}引分)")
    print(f"{args.deck_a}勝率: {a_wins / max(a_wins + b_wins, 1):.1%}")


if __name__ == '__main__':
    main()
