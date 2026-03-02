from rl_hybrid.sim.taker import execute_taker
from rl_hybrid.sim.maker import maker_transition
from rl_hybrid.sim.orders import MakerOrder
from rl_hybrid.sim.pnl import terminal_liquidation


def test_taker_exec():
    row = {'UP_ask':0.55,'UP_bid':0.54}
    filled, side, qty, px, fee = execute_taker(1,row,1.0,2.0)
    assert filled and side == 'UP' and qty == 1 and px > 0.55 and fee > 0


def test_maker_regimes():
    row = {'UP_spread':0.01,'UP_imb_1pp':0.0,'UP_holeBid':0.0,'quoteRate':0.1,'time_left_frac':0.8}
    od = MakerOrder(side='UP', is_bid=True, px=0.5, qty=1)
    for rg in ['optimistic','base','pessimistic']:
        st, _ = maker_transition(od, row, rg)
        assert st in {'pending','stale','fill_favorable','fill_adverse'}


def test_terminal_reward():
    assert terminal_liquidation(-0.5, 1, 0, 'UP') == 0.5
