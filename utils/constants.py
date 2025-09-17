# URLs
TOP10_HISTORIC_PLAYERS_URL = "https://www.chess.com/article/view/best-chess-players-of-all-time"
HISTORIC_PLAYER_GAMES_URL = "https://www.pgnmentor.com/files.html"
LCZERO_TEST60_URL = 'https://storage.lczero.org/files/training_pgns/test60/'

# Character sets for vision and chat transformer vocabs
PIECE_CHARS = "♔♕♖♗♘♙⭘♟♞♝♜♛♚"
PGN_CHARS = " #+-./0123456789:=BKLNOQRabcdefghx{}*"

# Path to stockfish binary used for engine
import os
import shutil

def _detect_stockfish_path() -> str:
    candidates = []
    # Local utils directory
    here = os.path.dirname(__file__)
    candidates.append(os.path.join(here, 'stockfish'))
    # Repo root utils directory (in case of different invocation path)
    repo_root = os.path.dirname(os.path.dirname(__file__))
    candidates.append(os.path.join(repo_root, 'utils', 'stockfish'))
    # Environment override
    env_path = os.environ.get('STOCKFISH_PATH')
    if env_path:
        candidates.insert(0, env_path)
    # System PATH
    which = shutil.which('stockfish')
    if which:
        candidates.append(which)
    for path in candidates:
        if path and os.path.exists(path):
            return path
    # Fallback to name hoping it is on PATH
    return 'stockfish'

STOCKFISH_PATH = _detect_stockfish_path()