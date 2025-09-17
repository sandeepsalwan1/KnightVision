import argparse
import io
import os
import chess
import chess.pgn
from typing import Dict

from chess_gameplay import Agent, play_game
from model import Model
import yaml

from engine.search import SearchAgent, SearchConfig, ModelEvaluator, MaterialEvaluator


def make_agent(kind: str, model: Model, depth: int, time_sec: float) -> Agent:
    kind = kind.lower()
    if kind == 'random':
        return Agent(model=None)
    if kind == 'model':
        return Agent(model=model)
    if kind == 'search':
        search_agent = SearchAgent(evaluator=MaterialEvaluator(), config=SearchConfig(max_depth=depth, time_limit_sec=time_sec))
        # Wrap to conform to Agent API
        class Wrapper:
            def score(self, pgn: str, move: str) -> float:
                # Not used by Agent when select_move overridden, but kept for compatibility
                return 0.0
            def select_move(self, pgn, legal_moves):
                return search_agent.select_move(pgn, legal_moves)
        return Agent(model=Wrapper())
    if kind in ('search+model', 'model+search'):
        evaluator = ModelEvaluator(model)
        search_agent = SearchAgent(evaluator=evaluator, config=SearchConfig(max_depth=depth, time_limit_sec=time_sec))
        class Wrapper:
            def score(self, pgn: str, move: str) -> float:
                return 0.0
            def select_move(self, pgn, legal_moves):
                return search_agent.select_move(pgn, legal_moves)
        return Agent(model=Wrapper())
    raise ValueError(f"Unknown agent kind: {kind}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a quick chess game between agents.")
    parser.add_argument('--white', default='model', help='random | model | search | search+model')
    parser.add_argument('--black', default='model', help='random | model | search | search+model')
    parser.add_argument('--max-moves', type=int, default=40)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--time', type=float, default=1.0, help='Time limit per search in seconds')
    parser.add_argument('--render', type=str, default=None, help='Path to save board images per move')
    args = parser.parse_args()

    # Load model config and checkpoint if available
    model_config = {}
    if os.path.exists('model_config.yaml'):
        model_config = yaml.safe_load(open('model_config.yaml'))
    model = Model(**model_config) if hasattr(Model, '__call__') else None
    if model and os.path.exists('checkpoint.pt'):
        import torch
        ckpt = torch.load('checkpoint.pt', map_location='cpu')
        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        try:
            model.load_state_dict(state_dict)
        except Exception:
            pass

    agents: Dict[str, Agent] = {
        'white': make_agent(args.white, model, args.depth, args.time),
        'black': make_agent(args.black, model, args.depth, args.time),
    }
    teams = {'white': f'White({args.white})', 'black': f'Black({args.black})'}

    result = play_game(agents=agents, teams=teams, max_moves=args.max_moves, min_seconds_per_move=0.0, verbose=True, poseval=False, image_path=args.render)
    print({k: v['points'] for k, v in result.items() if k in ('white','black')})

