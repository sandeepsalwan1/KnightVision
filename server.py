import os
import io
import uuid
from typing import Dict, List, Optional

import chess
import chess.svg
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chess_gameplay import Agent, sans_to_pgn
from model import Model
import yaml


class NewGameRequest(BaseModel):
    human_color: str = "white"  # "white" or "black"


class MoveRequest(BaseModel):
    game_id: str
    move_san: str


class Game:
    def __init__(self, human_color: str, agent: Agent):
        self.human_color: str = human_color
        self.agent: Agent = agent
        self.board: chess.Board = chess.Board()
        self.move_sans: List[str] = []

    @property
    def pgn(self) -> str:
        return sans_to_pgn(self.move_sans)


def load_model_if_available() -> Optional[Model]:
    model_config = {}
    if os.path.exists("model_config.yaml"):
        with open("model_config.yaml", "r") as f:
            model_config = yaml.safe_load(f) or {}

    try:
        model = Model(**model_config)  # type: ignore[arg-type]
    except Exception:
        return None

    # Load checkpoint if present
    ckpt_path = "checkpoint.pt"
    if os.path.exists(ckpt_path):
        try:
            import torch

            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            try:
                model.load_state_dict(state_dict)
            except Exception:
                pass
            try:
                model.eval()
            except Exception:
                pass
        except Exception:
            # Fallback to uninitialized model
            pass
    return model


def select_ai_move(game: Game) -> Optional[str]:
    if game.board.is_game_over():
        return None
    legal_moves = list(game.board.legal_moves)
    legal_sans = [game.board.san(m) for m in legal_moves]
    move_san = game.agent.select_move(game.pgn, legal_sans)
    try:
        game.board.push_san(move_san)
        game.move_sans.append(move_san)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AI produced invalid move: {move_san}: {exc}")
    return move_san


def serialize_state(game: Game) -> Dict:
    outcome = None
    if game.board.is_game_over():
        try:
            outcome_obj = game.board.outcome()
            if outcome_obj is not None:
                if outcome_obj.winner is None:
                    outcome = "draw"
                else:
                    outcome = "white" if outcome_obj.winner else "black"
        except Exception:
            outcome = "over"

    legal_sans: List[str] = []
    if not game.board.is_game_over():
        legal_sans = [game.board.san(m) for m in game.board.legal_moves]

    return {
        "fen": game.board.fen(),
        "turn": "white" if game.board.turn else "black",
        "pgn": game.pgn,
        "legal_moves": legal_sans,
        "is_game_over": game.board.is_game_over(),
        "outcome": outcome,
    }


app = FastAPI(title="KnightVision - Play the Model")

# Allow local dev from file:// or localhost origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static UI
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR, html=True), name="static")


# Global model and simple in-memory game store
GLOBAL_MODEL: Optional[Model] = load_model_if_available()
GAMES: Dict[str, Game] = {}


@app.get("/")
def root() -> Response:
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return Response(content="<h1>KnightVision</h1><p>UI not found. Visit /docs for API.</p>", media_type="text/html")


@app.post("/api/new")
def new_game(req: NewGameRequest) -> Dict:
    human_color = req.human_color.lower()
    if human_color not in ("white", "black"):
        raise HTTPException(status_code=400, detail="human_color must be 'white' or 'black'")

    agent = Agent(model=GLOBAL_MODEL)
    game = Game(human_color=human_color, agent=agent)
    game_id = str(uuid.uuid4())
    GAMES[game_id] = game

    # If AI plays white and human is black, make the first move
    if human_color == "black":
        select_ai_move(game)

    state = serialize_state(game)
    return {"game_id": game_id, **state}


@app.get("/api/state")
def get_state(game_id: str = Query(...)) -> Dict:
    game = GAMES.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="game_id not found")
    return serialize_state(game)


@app.get("/api/board.svg")
def board_svg(game_id: str = Query(...)) -> Response:
    game = GAMES.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="game_id not found")
    orientation = chess.WHITE if game.human_color == "white" else chess.BLACK
    last_move = game.board.peek() if game.move_sans else None
    svg = chess.svg.board(
        game.board,
        orientation=orientation,
        lastmove=last_move,
        coordinates=True,
        size=720,
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/api/move")
def make_move(req: MoveRequest) -> Dict:
    game = GAMES.get(req.game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="game_id not found")

    # Validate turn
    expected_turn = game.human_color
    current_turn = "white" if game.board.turn else "black"
    if expected_turn != current_turn:
        raise HTTPException(status_code=400, detail=f"Not your turn. Current turn: {current_turn}")

    # Attempt to push user's SAN move
    try:
        game.board.push_san(req.move_san)
        game.move_sans.append(req.move_san)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid move: {req.move_san}: {exc}")

    # If game over after human move, return
    if game.board.is_game_over():
        return serialize_state(game)

    # AI reply
    ai_move = select_ai_move(game)
    _ = ai_move  # not used in response explicitly; state includes last move
    return serialize_state(game)


# Simple PGN export if needed
@app.get("/api/pgn")
def export_pgn(game_id: str = Query(...)) -> Response:
    game = GAMES.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="game_id not found")
    return Response(content=game.pgn, media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)


