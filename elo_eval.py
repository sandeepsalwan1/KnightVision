import argparse
import io
import os
from typing import Dict, Tuple

import chess
import chess.pgn

from chess_gameplay import Agent, play_game
from engine.search import SearchAgent, SearchConfig, ModelEvaluator, MaterialEvaluator
from model import Model
import yaml

try:
    from utils.constants import STOCKFISH_PATH
    from chess.engine import SimpleEngine, Limit
    _HAS_STOCKFISH = True
except Exception:
    _HAS_STOCKFISH = False


class _StockfishWrapper:
    def __init__(self, time_sec: float, depth: int):
        if not _HAS_STOCKFISH:
            raise RuntimeError("Stockfish not available in this environment.")
        self.time_sec = max(0.0, float(time_sec))
        self.depth = int(depth) if depth and depth > 0 else None
        self.engine = SimpleEngine.popen_uci(STOCKFISH_PATH)

    def __del__(self):  # not guaranteed, but best-effort
        try:
            self.engine.quit()
        except Exception:
            pass

    def select_move(self, pgn: str, legal_move_sans):
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = chess.Board()
        if game is not None:
            for past_move in list(game.mainline_moves()):
                board.push(past_move)
        limit = Limit(time=self.time_sec) if self.depth is None else Limit(depth=self.depth)
        res = self.engine.play(board, limit)
        move = res.move
        return board.san(move)


def _make_agent(kind: str, model: Model, depth: int, time_sec: float) -> Agent:
    kind = (kind or '').lower()
    if kind == 'random':
        return Agent(model=None)
    if kind == 'model':
        return Agent(model=model)
    if kind == 'search':
        search_agent = SearchAgent(evaluator=MaterialEvaluator(), config=SearchConfig(max_depth=depth, time_limit_sec=time_sec))
        class Wrapper:
            def select_move(self, pgn, legal_moves):
                return search_agent.select_move(pgn, legal_moves)
        return Agent(model=Wrapper())
    if kind in ('search+model', 'model+search'):
        evaluator = ModelEvaluator(model)
        search_agent = SearchAgent(evaluator=evaluator, config=SearchConfig(max_depth=depth, time_limit_sec=time_sec))
        class Wrapper:
            def select_move(self, pgn, legal_moves):
                return search_agent.select_move(pgn, legal_moves)
        return Agent(model=Wrapper())
    if kind == 'stockfish':
        if not _HAS_STOCKFISH:
            raise RuntimeError("Stockfish is not available. Install a binary or set STOCKFISH_PATH.")
        wrapper = _StockfishWrapper(time_sec=time_sec, depth=depth)
        return Agent(model=wrapper)
    raise ValueError(f"Unknown agent kind: {kind}")


def _load_model() -> Model:
    model_config = {}
    if os.path.exists('model_config.yaml'):
        model_config = yaml.safe_load(open('model_config.yaml'))
    model = Model(**model_config)
    if os.path.exists('checkpoint.pt'):
        import torch
        ckpt = torch.load('checkpoint.pt', map_location='cpu')
        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        try:
            model.load_state_dict(state_dict)
        except Exception:
            pass
    try:
        model.eval()
    except Exception:
        pass
    return model


def _score_from_points(points: float) -> float:
    # Map {-1, 0, +1} -> {0.0, 0.5, 1.0}
    return (float(points) + 1.0) / 2.0


def _play_and_score(white_kind: str, black_kind: str, depth: int, time_sec: float) -> Tuple[float, Dict]:
    model = _load_model()
    agents: Dict[str, Agent] = {
        'white': _make_agent(white_kind, model, depth, time_sec),
        'black': _make_agent(black_kind, model, depth, time_sec),
    }
    teams = {'white': f'White({white_kind})', 'black': f'Black({black_kind})'}
    result = play_game(agents=agents, teams=teams, max_moves=200, min_seconds_per_move=0.0, verbose=False, poseval=False, image_path=None)
    white_points = result['white']['points']
    black_points = result['black']['points']
    white_score = _score_from_points(white_points)
    black_score = _score_from_points(black_points)
    return white_score, {'white_points': white_points, 'black_points': black_points}


def _estimate_elo_delta(score_mean: float, n_games: int) -> Tuple[float, float]:
    import math
    score = min(max(score_mean, 1e-6), 1 - 1e-6)
    delta = 400.0 * math.log10(score / (1.0 - score))
    # Standard error of mean
    se = (score * (1.0 - score) / max(n_games, 1)) ** 0.5
    # Derivative of delta w.r.t score
    ddelta_dscore = 400.0 / math.log(10) * (1.0 / score + 1.0 / (1.0 - score))
    se_delta = abs(ddelta_dscore) * se
    ci95 = 1.96 * se_delta
    return delta, ci95


def main():
    parser = argparse.ArgumentParser(description="Estimate Elo by head-to-head matches.")
    parser.add_argument('--agent-a', default='model', help='random | model | search | search+model | stockfish')
    parser.add_argument('--agent-b', default='search', help='random | model | search | search+model | stockfish')
    parser.add_argument('--games', type=int, default=20, help='Total number of games (will alternate colors)')
    parser.add_argument('--depth', type=int, default=3, help='Search/Stockfish depth (if used)')
    parser.add_argument('--time', type=float, default=0.5, help='Time limit for search/stockfish (sec)')
    parser.add_argument('--baseline-rating', type=float, default=None, help='Optional baseline rating for agent-b to compute absolute Elo for agent-a')
    args = parser.parse_args()

    a_wins = 0
    a_draws = 0
    a_losses = 0
    a_scores = []

    for i in range(args.games):
        # Alternate colors for fairness
        if i % 2 == 0:
            white_kind, black_kind = args.agent_a, args.agent_b
            a_is_white = True
        else:
            white_kind, black_kind = args.agent_b, args.agent_a
            a_is_white = False

        score_white, _meta = _play_and_score(white_kind, black_kind, args.depth, args.time)
        # A's score this game
        a_score = score_white if a_is_white else (1.0 - score_white)
        a_scores.append(a_score)
        if a_score > 0.5:
            a_wins += 1
        elif a_score < 0.5:
            a_losses += 1
        else:
            a_draws += 1

        print(f"Game {i+1}/{args.games}: A({args.agent_a}) vs B({args.agent_b}) -> score {a_score:.1f}")

    mean_score = sum(a_scores) / max(len(a_scores), 1)
    delta, ci95 = _estimate_elo_delta(mean_score, len(a_scores))

    print("\nResults:")
    print(f"A({args.agent_a}) vs B({args.agent_b}) over {len(a_scores)} games")
    print(f"Record: {a_wins} wins, {a_draws} draws, {a_losses} losses")
    print(f"Mean score: {mean_score:.3f}")
    print(f"Elo difference (A - B): {delta:.1f} ± {ci95:.1f} (95% CI)")
    if args.baseline_rating is not None:
        absolute = args.baseline_rating + delta
        print(f"Estimated absolute Elo for A: {absolute:.0f} ± {ci95:.1f}")


if __name__ == '__main__':
    main()


