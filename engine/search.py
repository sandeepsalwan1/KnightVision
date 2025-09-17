import io
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, List

import chess
import chess.pgn


def _default_move_order(board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
    def score_move(m: chess.Move) -> int:
        score = 0
        if board.is_capture(m):
            score += 1000
        if board.is_en_passant(m):
            score += 500
        if board.gives_check(m):
            score += 300
        if m.promotion:
            score += 800 + m.promotion
        return score

    return sorted(moves, key=score_move, reverse=True)


@dataclass
class SearchConfig:
    max_depth: int = 3
    time_limit_sec: float = 1.0
    use_iterative_deepening: bool = True
    quiescence_depth: int = 0
    transposition_table: bool = True
    move_ordering: bool = True


class Evaluator:
    def evaluate(self, board: chess.Board) -> float:
        raise NotImplementedError


class MaterialEvaluator(Evaluator):
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0,
    }

    def evaluate(self, board: chess.Board) -> float:
        # Positive means good for side to move.
        score = 0
        for piece_type, value in self.PIECE_VALUES.items():
            score += value * (len(board.pieces(piece_type, board.turn)) - len(board.pieces(piece_type, not board.turn)))
        return float(score)


class ModelEvaluator(Evaluator):
    def __init__(self, model):
        self.model = model
        try:
            self.model.eval()
        except Exception:
            pass

    def evaluate(self, board: chess.Board) -> float:
        # Expect model to return score from the perspective of the side to move.
        # Prefer a direct board evaluation method if present.
        if hasattr(self.model, "evaluate_board"):
            value = self.model.evaluate_board(board)
            return float(value)

        # Fallback: reconstruct a PGN and score a null move equivalent by averaging moves.
        # Better: directly call forward using model.encode_board.
        try:
            from model import encode_board  # type: ignore
            import torch

            with torch.no_grad():
                board_tensor = torch.tensor(encode_board(board)).unsqueeze(0)
                value = self.model(board_tensor).item()
                return float(value)
        except Exception:
            # Last resort: simple material
            return MaterialEvaluator().evaluate(board)


class TranspositionTable:
    def __init__(self) -> None:
        self._table: Dict[str, Tuple[int, float, Optional[chess.Move]]] = {}

    def probe(self, board: chess.Board, depth: int) -> Optional[Tuple[int, float, Optional[chess.Move]]]:
        key = board.board_fen() + (" w" if board.turn else " b")
        return self._table.get(key)

    def store(self, board: chess.Board, depth: int, value: float, move: Optional[chess.Move]) -> None:
        key = board.board_fen() + (" w" if board.turn else " b")
        self._table[key] = (depth, value, move)


class AlphaBetaSearcher:
    def __init__(self, evaluator: Evaluator, config: Optional[SearchConfig] = None) -> None:
        self.evaluator = evaluator
        self.config = config or SearchConfig()
        self.tt = TranspositionTable() if self.config.transposition_table else None
        self.start_time: float = 0.0

    def _time_up(self) -> bool:
        if self.config.time_limit_sec <= 0:
            return False
        return (time.perf_counter() - self.start_time) >= self.config.time_limit_sec

    def search(self, board: chess.Board) -> chess.Move:
        self.start_time = time.perf_counter()
        best_move: Optional[chess.Move] = None
        max_depth = self.config.max_depth

        if self.config.use_iterative_deepening:
            for depth in range(1, max_depth + 1):
                if self._time_up():
                    break
                value, move = self._alphabeta_root(board, depth)
                if move is not None:
                    best_move = move
        else:
            _, best_move = self._alphabeta_root(board, max_depth)

        # Fallback if no move found (shouldn't happen)
        return best_move or next(iter(board.legal_moves))

    def _alphabeta_root(self, board: chess.Board, depth: int) -> Tuple[float, Optional[chess.Move]]:
        alpha = float("-inf")
        beta = float("inf")
        best_value = float("-inf")
        best_move: Optional[chess.Move] = None

        legal_moves = list(board.legal_moves)
        if self.config.move_ordering:
            legal_moves = _default_move_order(board, legal_moves)

        for move in legal_moves:
            if self._time_up():
                break
            board.push(move)
            value = -self._alphabeta(board, depth - 1, -beta, -alpha)
            board.pop()
            if value > best_value:
                best_value = value
                best_move = move
            if value > alpha:
                alpha = value
        return best_value, best_move

    def _alphabeta(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        if self._time_up():
            return self.evaluator.evaluate(board)

        if depth <= 0:
            return self._quiescence(board, alpha, beta, self.config.quiescence_depth)

        if self.tt is not None:
            hit = self.tt.probe(board, depth)
            if hit is not None:
                stored_depth, stored_value, _ = hit
                if stored_depth >= depth:
                    return stored_value

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            if board.is_checkmate():
                # Large negative for side to move in checkmate
                return -float(1e6)
            return 0.0

        if self.config.move_ordering:
            legal_moves = _default_move_order(board, legal_moves)

        value = float("-inf")
        for move in legal_moves:
            board.push(move)
            score = -self._alphabeta(board, depth - 1, -beta, -alpha)
            board.pop()
            if score > value:
                value = score
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break

        if self.tt is not None:
            self.tt.store(board, depth, value, None)
        return value

    def _quiescence(self, board: chess.Board, alpha: float, beta: float, q_depth: int) -> float:
        stand_pat = self.evaluator.evaluate(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        if q_depth <= 0:
            return stand_pat

        capture_moves = [m for m in board.legal_moves if board.is_capture(m)]
        if self.config.move_ordering:
            capture_moves = _default_move_order(board, capture_moves)

        for move in capture_moves:
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, q_depth - 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha


class SearchAgent:
    def __init__(self, evaluator: Optional[Evaluator] = None, config: Optional[SearchConfig] = None):
        self.searcher = AlphaBetaSearcher(evaluator or MaterialEvaluator(), config)

    def select_move(self, pgn: str, legal_move_sans: List[str]) -> str:
        # Reconstruct board from PGN string
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = chess.Board()
        if game is not None:
            for past_move in list(game.mainline_moves()):
                board.push(past_move)

        best_move = self.searcher.search(board)
        return board.san(best_move)

