import os
import pytest
import chess

from model import Model
from chess_gameplay import Agent, play_game


def test_model_instantiates():
    m = Model()
    assert hasattr(m, 'score')


def test_play_quick_game():
    m0 = Model()
    a0, a1 = Agent(m0), Agent(m0)
    result = play_game(
        agents={'white': a0, 'black': a1},
        teams={'white': 'W', 'black': 'B'},
        max_moves=4,
        min_seconds_per_move=0.0,
        verbose=False,
        poseval=False,
        image_path=None,
    )
    assert 'white' in result and 'black' in result

