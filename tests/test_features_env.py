import pandas as pd

from rl_hybrid.features.engineering import build_features
from rl_hybrid.features.episodes import build_episodes
from rl_hybrid.env.market_env import HybridMarketEnv


def _df():
    return pd.DataFrame([
        {"type":"tick","ts":1,"asset":"BTC","cycle":"c1","winner":"UP","quoteRate":0.1,"UP_bid":0.45,"UP_ask":0.46,"UP_spread":0.01,"DOWN_bid":0.54,"DOWN_ask":0.55,"DOWN_spread":0.01,"UP_bidDepth_1pp":100,"DOWN_bidDepth_1pp":120},
        {"type":"tick","ts":2,"asset":"BTC","cycle":"c1","winner":"UP","quoteRate":0.2,"UP_bid":0.47,"UP_ask":0.48,"UP_spread":0.01,"DOWN_bid":0.52,"DOWN_ask":0.53,"DOWN_spread":0.01,"UP_bidDepth_1pp":80,"DOWN_bidDepth_1pp":130},
    ])


def test_features_no_leakage_like():
    f = build_features(_df(), [2])
    assert 'implied_up_mean_2' in f.columns
    assert f.iloc[0]['implied_up_std_2'] == 0


def test_env_transitions_and_inventory_limit():
    f = build_features(_df())
    eps = build_episodes(f)
    env = HybridMarketEnv(eps[0], ['implied_up','quoteRate','time_left_frac'], {'maker_regime':'base', 'max_inventory_per_side': 0})
    o,_ = env.reset()
    assert o.shape[0] == 11
    _,r,_,_,info = env.step(1)
    assert isinstance(r, float)
    assert info['pos_up'] == 0
