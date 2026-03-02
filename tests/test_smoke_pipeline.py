import pandas as pd
from rl_hybrid.features.engineering import build_features
from rl_hybrid.features.episodes import build_episodes, temporal_split
from rl_hybrid.train.supervised_pipeline import train_supervised
from rl_hybrid.train.rl_pipeline import train_dqn
from rl_hybrid.eval.backtest import run_backtest


def _dataset(n=20):
    rows=[]
    for c in range(n):
        for t in range(3):
            rows.append({"type":"tick","ts":t,"asset":"BTC","cycle":f'c{c}',"winner":'UP' if c%2==0 else 'DOWN',"quoteRate":0.1+t*0.01,
                         "UP_bid":0.45+t*0.01,"UP_ask":0.46+t*0.01,"UP_spread":0.01,
                         "DOWN_bid":0.54-t*0.01,"DOWN_ask":0.55-t*0.01,"DOWN_spread":0.01,
                         "UP_bidDepth_1pp":100,"DOWN_bidDepth_1pp":110})
    return build_features(pd.DataFrame(rows))


def test_train_and_backtest_smoke(tmp_path):
    df = _dataset()
    eps = build_episodes(df)
    tr,va,te = temporal_split(eps)
    cols=['implied_up','thinness','spread_asym','depth_asym','quoteRate','time_left_frac','d_UP_bid','d_DOWN_bid']
    train_supervised(pd.concat([e.data.iloc[[-1]] for e in tr]), pd.concat([e.data.iloc[[-1]] for e in va]), pd.concat([e.data.iloc[[-1]] for e in te]), cols, tmp_path/'sup')
    cfg={'train_episodes':4,'batch_size':4,'target_update':5,'maker_regime':'base'}
    train_dqn(tr, cols, cfg, tmp_path/'rl')
    rep = run_backtest(te, cols, {'max_active_orders':4}, str(tmp_path/'rl'/'best_dqn.pt'), tmp_path/'bt')
    assert 'base' in rep
