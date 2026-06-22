"""
自己対戦データ収集スクリプト v2
v10ヒューリスティックで両者を動かし、(盤面特徴量, 勝敗ラベル, 重み) を収集する。
複数デッキアーキタイプの対戦で多様なデータを収集する。

使い方:
    python collect_data.py [--games 20000]
"""
import argparse
import random
import numpy as np

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, OptionType, SelectContext, AreaType, all_card_data, all_attack

# ── デッキ定義 ────────────────────────────────────────────────────────────

# 1. Fighting: Mega Lucario ex (v10 / 私たちのデッキ)
DECK_FIGHTING = (
    [673] * 2 + [674] * 2 + [675] * 1 + [676] * 3 + [677] * 4 + [678] * 4
  + [1102] * 4 + [1123] * 2 + [1141] * 4 + [1142] * 4 + [1152] * 4
  + [1159] * 1 + [1182] * 2 + [1192] * 4 + [1227] * 4 + [1252] * 2
  + [6] * 13
)

# 2. Psychic: Alakazam / Dudunsparce (Soraice deck)
DECK_PSYCHIC_ALAKAZAM = (
    [65] * 4 + [66] * 4 + [741] * 4 + [742] * 4 + [743] * 3
  + [1079] * 3 + [1081] * 3 + [1086] * 4 + [1097] * 1 + [1129] * 1
  + [1146] * 1 + [1152] * 4 + [1159] * 1 + [1182] * 3 + [1184] * 1
  + [1225] * 4 + [1231] * 4 + [1264] * 4
  + [5] * 3 + [19] * 4
)

# 3. Psychic: Alakazam / Dunsparce / Fezandipiti complex (ulaph4 deck)
DECK_PSYCHIC_COMPLEX = (
    [66] * 3 + [140] * 1 + [305] * 3 + [343] * 1
  + [741] * 4 + [742] * 4 + [743] * 3
  + [1079] * 3 + [1081] * 2 + [1086] * 4 + [1097] * 1 + [1129] * 1
  + [1146] * 1 + [1152] * 4 + [1182] * 3 + [1184] * 1 + [1186] * 1
  + [1225] * 4 + [1231] * 4 + [1248] * 4 + [1264] * 1
  + [5] * 2 + [13] * 1 + [19] * 4
)

# 4. Grass/Psychic: Dudunsparce / 878-line (persn deck)
DECK_GRASS_SUSTAIN = (
    [65] * 4 + [66] * 3 + [304] * 2 + [878] * 4 + [879] * 2
  + [1086] * 4 + [1097] * 3 + [1115] * 3 + [1122] * 4 + [1152] * 4
  + [1171] * 4 + [1182] * 2 + [1194] * 2 + [1210] * 2 + [1227] * 4 + [1255] * 4
  + [11] * 4 + [12] * 1 + [19] * 4
)

# 5. Grass/Fire: 352-354 line (llkarill deck)
DECK_GRASS_FIRE = (
    [65] * 4 + [66] * 4 + [352] * 4 + [353] * 4 + [354] * 4
  + [1079] * 2 + [1086] * 4 + [1114] * 1 + [1121] * 4 + [1122] * 4
  + [1129] * 1 + [1152] * 4 + [1182] * 2 + [1215] * 4 + [1224] * 2 + [1227] * 4
  + [2] * 7 + [12] * 1
)

# 6. Lightning/Water: YT deck (22 energy ramp)
DECK_LIGHTNING = (
    [265] * 3 + [268] * 3 + [269] * 3 + [270] * 3 + [271] * 3
  + [1086] * 3 + [1097] * 2 + [1110] * 1 + [1118] * 1 + [1121] * 3
  + [1152] * 2 + [1227] * 4 + [1233] * 4 + [1254] * 3
  + [4] * 22
)

DECKS = [
    DECK_FIGHTING, DECK_FIGHTING, DECK_FIGHTING,   # 43%
    DECK_PSYCHIC_ALAKAZAM,                          # 14%
    DECK_PSYCHIC_COMPLEX,                           # 14%
    DECK_GRASS_SUSTAIN,                             # 14%
    DECK_GRASS_FIRE,                                # 7.5%
    DECK_LIGHTNING,                                 # 7.5%
]

FEATURE_DIM = 100


# ── カードDB構築（起動時1回だけ） ─────────────────────────────────────────

def _build_card_db():
    db = {}
    try:
        for c in all_card_data():
            db[c.cardId] = c
    except Exception:
        pass
    return db

def _build_attack_db():
    db = {}
    try:
        for a in all_attack():
            db[a.attackId] = a
    except Exception:
        pass
    return db

CARD_DB   = _build_card_db()
ATTACK_DB = _build_attack_db()


# ── カードID分類ヘルパー ──────────────────────────────────────────────────

def _is_pokemon(cid):  return 21 <= cid <= 999
def _is_trainer(cid):  return cid >= 1000
def _is_energy(cid):   return cid <= 20

def _card_composition(cards):
    """カードリスト → (ポケモン数, トレーナー数, エネルギー数)"""
    if not cards:
        return 0, 0, 0
    poke = sum(1 for c in cards if hasattr(c, 'id') and _is_pokemon(c.id))
    trainer = sum(1 for c in cards if hasattr(c, 'id') and _is_trainer(c.id))
    energy = sum(1 for c in cards if hasattr(c, 'id') and _is_energy(c.id))
    return poke, trainer, energy


# ── 戦術的特徴量ヘルパー ──────────────────────────────────────────────────

def _get_card_data(pokemon):
    """ポケモンオブジェクト → CardData (なければNone)"""
    if pokemon is None:
        return None
    return CARD_DB.get(pokemon.id)

def _best_attack_damage(pokemon):
    """ポケモンの最大攻撃力を返す"""
    cd = _get_card_data(pokemon)
    if cd is None:
        return 0
    best = 0
    for aid in (cd.attacks or []):
        atk = ATTACK_DB.get(aid)
        if atk and atk.damage > best:
            best = atk.damage
    return best

def _can_afford_attack(attached_energies, required_energies):
    """エネルギータイプを考慮した攻撃コスト判定。COLORLESS(0)は任意タイプで支払可。"""
    from collections import Counter
    req = Counter(required_energies)
    att = Counter(attached_energies)
    colorless_needed = req.pop(0, 0)  # EnergyType.COLORLESS = 0
    for etype, count in req.items():
        if att[etype] < count:
            return False
        att[etype] -= count
    return sum(att.values()) >= colorless_needed

def _can_attack(pokemon):
    """現在のエネルギーで攻撃可能か（タイプ一致チェック付き）"""
    cd = _get_card_data(pokemon)
    if cd is None or pokemon is None:
        return False
    for aid in (cd.attacks or []):
        atk = ATTACK_DB.get(aid)
        if atk and _can_afford_attack(pokemon.energies, atk.energies):
            return True
    return False

def _affordable_damage(pokemon):
    """支払い可能な攻撃の最大ダメージ（タイプ一致チェック付き）"""
    cd = _get_card_data(pokemon)
    if cd is None or pokemon is None:
        return 0
    best = 0
    for aid in (cd.attacks or []):
        atk = ATTACK_DB.get(aid)
        if atk and _can_afford_attack(pokemon.energies, atk.energies) and atk.damage > best:
            best = atk.damage
    return best

def _weakness_match(attacker, defender):
    """attackerのタイプがdefenderの弱点と一致するか"""
    acd = _get_card_data(attacker)
    dcd = _get_card_data(defender)
    if acd is None or dcd is None or dcd.weakness is None:
        return False
    return acd.energyType == dcd.weakness

def _max_evolution_stats(pokemon):
    """最終進化形のmaxHPと最大攻撃力を返す（ふしぎのアメ込み）"""
    cd = _get_card_data(pokemon)
    if cd is None:
        return 0, 0
    best_hp  = cd.hp
    best_dmg = _best_attack_damage(pokemon)
    name = cd.name
    for _ in range(3):  # basic→stage1→stage2→mega (最大3段階)
        found = False
        for c in CARD_DB.values():
            if c.evolvesFrom == name:
                best_hp = max(best_hp, c.hp)
                for aid in (c.attacks or []):
                    atk = ATTACK_DB.get(aid)
                    if atk:
                        best_dmg = max(best_dmg, atk.damage)
                name = c.name
                found = True
                break
        if not found:
            break
    return best_hp, best_dmg


def _prize_risk(pokemon):
    """撃破時に相手が得るサイド枚数 (0=不在/None, 1=通常, 2=ex, 3=megaEx)"""
    if pokemon is None:
        return 0
    cd = _get_card_data(pokemon)
    if cd is None:
        return 1  # 不明でもポケモンなら最低1枚
    if cd.megaEx:
        return 3
    if cd.ex:
        return 2
    return 1


# ── 特徴量抽出 ─────────────────────────────────────────────────────────────

def extract_features(obs):
    """
    現在の盤面を96次元のfloat32ベクトルに変換。

    内訳:
      自分アクティブ:  HP, エネ, maxHP, 進化, ツール, 状態×2             (7)
      自分ベンチ×5:    HP, エネ, maxHP, 進化                             (20)
      相手アクティブ:  HP, エネ, maxHP, 進化, 状態                        (5)
      相手ベンチ×5:    HP, エネ, maxHP                                    (15)
      グローバル:      サイド,手札,デッキ,捨札,ベンチ数,ターン等           (23)
      戦術:           攻撃力,攻撃可否,弱点,KO判定,サイドリスク,退却コスト  (11)
      戦略:           ベンチ攻撃準備,KO計画,ボス枚数,進化手札,相手進化     (15)
      予測:           進化ポテンシャル(HP,攻撃力)×自分相手                (4)
      合計 = 100
    """
    state = obs.current
    if state is None:
        return None

    our_idx = state.yourIndex
    opp_idx = 1 - our_idx
    us  = state.players[our_idx]
    opp = state.players[opp_idx]

    feats = []

    # --- 自分のアクティブ (7次元) ---
    a = us.active[0] if us.active else None
    feats += [
        a.hp / a.maxHp if (a and a.maxHp > 0) else 0.0,
        min(len(a.energies), 8) / 8.0 if a else 0.0,
        min(getattr(a, 'maxHp', 0), 350) / 350.0 if a else 0.0,
        1.0 if (a and getattr(a, 'preEvolution', None)) else 0.0,
        min(len(getattr(a, 'tools', None) or []), 2) / 2.0 if a else 0.0,
        1.0 if (getattr(us, 'asleep', False) or getattr(us, 'confused', False)
                or getattr(us, 'paralyzed', False)) else 0.0,
        1.0 if (getattr(us, 'poisoned', False)
                or getattr(us, 'burned', False)) else 0.0,
    ]

    # --- 自分のベンチ 5体 (20次元) ---
    bench = us.bench or []
    for i in range(5):
        b = bench[i] if i < len(bench) else None
        feats += [
            b.hp / b.maxHp if (b and b.maxHp > 0) else 0.0,
            min(len(b.energies), 8) / 8.0 if b else 0.0,
            min(getattr(b, 'maxHp', 0), 350) / 350.0 if b else 0.0,
            1.0 if (b and getattr(b, 'preEvolution', None)) else 0.0,
        ]

    # --- 相手のアクティブ (5次元) ---
    oa = opp.active[0] if opp.active else None
    feats += [
        oa.hp / oa.maxHp if (oa and oa.maxHp > 0) else 0.0,
        min(len(oa.energies), 8) / 8.0 if oa else 0.0,
        min(getattr(oa, 'maxHp', 0), 350) / 350.0 if oa else 0.0,
        1.0 if (oa and getattr(oa, 'preEvolution', None)) else 0.0,
        1.0 if (getattr(opp, 'asleep', False) or getattr(opp, 'confused', False)
                or getattr(opp, 'paralyzed', False) or getattr(opp, 'poisoned', False)
                or getattr(opp, 'burned', False)) else 0.0,
    ]

    # --- 相手のベンチ 5体 (15次元) ---
    opp_bench = opp.bench or []
    for i in range(5):
        ob = opp_bench[i] if i < len(opp_bench) else None
        feats += [
            ob.hp / ob.maxHp if (ob and ob.maxHp > 0) else 0.0,
            min(len(ob.energies), 8) / 8.0 if ob else 0.0,
            min(getattr(ob, 'maxHp', 0), 350) / 350.0 if ob else 0.0,
        ]

    # --- グローバル (23次元) ---
    our_hand = getattr(us, 'hand', None) or []
    h_poke, h_trainer, h_energy = _card_composition(our_hand)

    our_discard = us.discard or []
    opp_discard = opp.discard or []
    d_poke, d_trainer, d_energy = _card_composition(our_discard)
    od_poke, od_trainer, od_energy = _card_composition(opp_discard)

    feats += [
        len(us.prize)  / 6.0,                                          # 1
        len(opp.prize) / 6.0,                                          # 2
        min(getattr(us,  'handCount', 0), 10) / 10.0,                  # 3
        min(getattr(opp, 'handCount', 0), 10) / 10.0,                  # 4
        getattr(us,  'deckCount', 0) / 60.0,                           # 5
        getattr(opp, 'deckCount', 0) / 60.0,                           # 6
        len(our_discard) / 60.0,                                        # 7
        len(opp_discard) / 60.0,                                        # 8
        len(bench)     / 5.0,                                           # 9
        len(opp_bench) / 5.0,                                           # 10
        1.0 if getattr(state, 'energyAttached', False) else 0.0,        # 11
        1.0 if getattr(state, 'stadium', None) else 0.0,                # 12
        min(getattr(state, 'turn', 0), 30) / 30.0,                     # 13
        1.0 if getattr(state, 'firstPlayer', 0) == our_idx else 0.0,   # 14
        1.0 if getattr(state, 'supporterPlayed', False) else 0.0,      # 15
        1.0 if getattr(state, 'retreated', False) else 0.0,            # 16
        min(h_poke,    10) / 10.0,                                         # 17
        min(h_trainer, 10) / 10.0,                                         # 18
        min(h_energy,  10) / 10.0,                                         # 19
        od_poke    / 30.0,                                              # 20
        od_trainer / 30.0,                                              # 21
        od_energy  / 30.0,                                              # 22
        d_energy   / 30.0,                                              # 23
    ]

    # --- 戦術的特徴量 (11次元) --- CardData参照
    our_dmg  = _affordable_damage(a)     # 自分が出せるダメージ
    opp_dmg  = _affordable_damage(oa)    # 相手が出せるダメージ
    opp_hp   = oa.hp if oa else 0
    our_hp   = a.hp  if a  else 0

    # 弱点考慮のダメージ (x2)
    our_eff = our_dmg * 2 if _weakness_match(a, oa) else our_dmg
    opp_eff = opp_dmg * 2 if _weakness_match(oa, a) else opp_dmg

    feats += [
        min(our_dmg, 350) / 350.0,                                      # 24: 攻撃力
        min(opp_dmg, 350) / 350.0,                                      # 25: 相手攻撃力
        1.0 if _can_attack(a) else 0.0,                                 # 26: 攻撃可能
        1.0 if _can_attack(oa) else 0.0,                                # 27: 相手攻撃可能
        1.0 if _weakness_match(a, oa) else 0.0,                         # 28: 弱点突ける
        1.0 if _weakness_match(oa, a) else 0.0,                         # 29: 弱点突かれる
        1.0 if (opp_hp > 0 and our_eff >= opp_hp) else 0.0,            # 30: KO可能！
        1.0 if (our_hp > 0 and opp_eff >= our_hp) else 0.0,            # 31: KOされうる
        _prize_risk(a)  / 3.0,                                          # 32: 自分サイドリスク
        _prize_risk(oa) / 3.0,                                          # 33: 相手サイド価値
        min(getattr(_get_card_data(a), 'retreatCost', 0), 5) / 5.0,  # 34: にげるコスト
    ]

    # --- 戦略的特徴量 (15次元) --- 人間が考える展開予測
    opp_all_pokemon = ([oa] if oa else []) + [ob for ob in (opp_bench or []) if ob]
    our_all_pokemon = ([a]  if a  else []) + [b  for b  in (bench or [])     if b ]

    # A. 自分ベンチの攻撃準備度 ×5  (35-39)
    for i in range(5):
        b = bench[i] if i < len(bench) else None
        feats.append(1.0 if _can_attack(b) else 0.0)

    # B. 勝利までの最短KO数 (40-41)
    our_remaining  = len(us.prize)  # 自分が取るべき残りサイド枚数
    opp_remaining  = len(opp.prize)
    # 相手ポケモンのサイド価値を降順ソート → 最短何体で取り切れるか
    opp_prize_vals = sorted([_prize_risk(p) for p in opp_all_pokemon], reverse=True)
    kos_needed = 0
    total_prizes = 0
    for pv in opp_prize_vals:
        if total_prizes >= our_remaining:
            break
        total_prizes += pv
        kos_needed += 1
    if total_prizes < our_remaining:
        kos_needed = 6  # 場のポケモンだけでは足りない
    feats.append(min(kos_needed, 6) / 6.0)

    our_prize_vals = sorted([_prize_risk(p) for p in our_all_pokemon], reverse=True)
    opp_kos_needed = 0
    total_prizes = 0
    for pv in our_prize_vals:
        if total_prizes >= opp_remaining:
            break
        total_prizes += pv
        opp_kos_needed += 1
    if total_prizes < opp_remaining:
        opp_kos_needed = 6
    feats.append(min(opp_kos_needed, 6) / 6.0)

    # C. ボスの司令カウント (42-43)  ID=1182
    our_boss_used = sum(1 for c in our_discard if hasattr(c, 'id') and c.id == 1182)
    opp_boss_used = sum(1 for c in opp_discard if hasattr(c, 'id') and c.id == 1182)
    feats.append(our_boss_used / 4.0)
    feats.append(opp_boss_used / 4.0)

    # D. 手札に進化カードがあるか (44) ※アクティブも含む
    field_names = set()
    for p in ([a] + list(bench)):
        if p is None:
            continue
        cd = _get_card_data(p)
        if cd:
            field_names.add(cd.name)
    has_evo_in_hand = 0.0
    for c in our_hand:
        if not hasattr(c, 'id'):
            continue
        cd = CARD_DB.get(c.id)
        if cd and cd.evolvesFrom and cd.evolvesFrom in field_names:
            has_evo_in_hand = 1.0
            break
    feats.append(has_evo_in_hand)

    # E. 相手ベンチの進化済みフラグ ×5 (45-49)
    for i in range(5):
        ob = opp_bench[i] if i < len(opp_bench) else None
        feats.append(1.0 if (ob and getattr(ob, 'preEvolution', None)) else 0.0)

    # F. 進化ポテンシャル (50-53) — ベンチの将来脅威度
    # 相手ベンチの最大進化ポテンシャル (maxHP, 攻撃力)
    opp_best_evo_hp, opp_best_evo_dmg = 0, 0
    for ob in (opp_bench or []):
        if ob is None:
            continue
        hp, dmg = _max_evolution_stats(ob)
        opp_best_evo_hp  = max(opp_best_evo_hp, hp)
        opp_best_evo_dmg = max(opp_best_evo_dmg, dmg)
    feats.append(min(opp_best_evo_hp,  350) / 350.0)
    feats.append(min(opp_best_evo_dmg, 350) / 350.0)

    # 自分ベンチの最大進化ポテンシャル (maxHP, 攻撃力)
    our_best_evo_hp, our_best_evo_dmg = 0, 0
    for b in (bench or []):
        if b is None:
            continue
        hp, dmg = _max_evolution_stats(b)
        our_best_evo_hp  = max(our_best_evo_hp, hp)
        our_best_evo_dmg = max(our_best_evo_dmg, dmg)
    feats.append(min(our_best_evo_hp,  350) / 350.0)
    feats.append(min(our_best_evo_dmg, 350) / 350.0)

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


MEGA_LUCARIO_ID = 678

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


def heuristic_action(obs, fallback_deck=None):
    if obs.select is None:
        return list(fallback_deck or DECK_FIGHTING)
    options = obs.select.option
    n       = len(options)
    scores  = _score_options(obs)
    desc    = sorted(range(n), key=lambda k: scores[k], reverse=True)
    return desc[:obs.select.maxCount]


# ── 1試合実行 ─────────────────────────────────────────────────────────────

def run_game():
    """
    ランダムに2つのデッキを選んで1試合を実行。
    Returns:
        records: list of (features, player_idx, step_idx)
        winner:  0 or 1  勝者プレイヤーインデックス (-1 = 未決)
    """
    deck0 = random.choice(DECKS)
    deck1 = random.choice(DECKS)
    try:
        obs_dict, _ = battle_start(list(deck0), list(deck1))
    except Exception:
        return [], -1

    records = []
    step_idx = 0

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
            return records, winner

        if obs.current and obs.select:
            feats = None
            try:
                feats = extract_features(obs)
            except Exception:
                pass
            if feats is not None:
                records.append((feats, obs.current.yourIndex, step_idx))
                step_idx += 1

        try:
            action   = heuristic_action(obs, fallback_deck=deck0)
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
    parser.add_argument('--games', type=int, default=20000)
    parser.add_argument('--out',   type=str, default='model')
    args = parser.parse_args()

    all_X, all_y, all_w = [], [], []
    skip = 0

    print(f"{args.games} 試合を実行中...")
    for i in range(args.games):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{args.games} (サンプル数: {len(all_X)}, スキップ: {skip})")

        records, winner = run_game()
        if winner < 0:
            skip += 1
            continue

        total_steps = len(records)
        for feats, player_idx, step_idx in records:
            label = 1.0 if player_idx == winner else 0.0
            # ゲーム段階重み: 序盤0.2 → 終盤1.0 (終盤ほど信頼度が高い)
            progress = step_idx / max(total_steps - 1, 1)
            weight = 0.2 + 0.8 * progress
            all_X.append(feats)
            all_y.append(label)
            all_w.append(weight)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    w = np.array(all_w, dtype=np.float32)

    import os
    os.makedirs(args.out, exist_ok=True)
    np.save(f'{args.out}/train_X.npy', X)
    np.save(f'{args.out}/train_y.npy', y)
    np.save(f'{args.out}/train_w.npy', w)

    print(f"\n完了: {len(all_X)} サンプル保存 → {args.out}/")
    print(f"勝率バランス: {y.mean():.3f} (0.5 に近いほど良い)")
    print(f"スキップ: {skip} 試合")


if __name__ == '__main__':
    main()
