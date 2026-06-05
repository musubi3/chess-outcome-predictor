from chess.pgn import Mainline
from chess import Move
from typing import Literal

def parse_time_control(tc_str: str) -> tuple[int, int]:
    '''
    Parses Lichess TimeControl string (e.g., '180+2') into 
    base seconds and increment.
    '''
    if tc_str in ['-', '?', ''] or '+' not in tc_str:
        return 0, 0
    
    try:
        base: int; inc: int
        base, inc = map(int, tc_str.split('+'))
        return base, inc
    except ValueError:
        return 0, 0
    
def parse_event(event_str: str, categories: list[str] = ['Bullet', 'Blitz', 'Rapid', 'Classical']) -> str | None:
    '''
    Parses Lichess Event string to determine speed category.
    '''
    for cat in categories:
        if cat in event_str:
            return cat
    return None

def parse_elo(w_elo_str: str, b_elo_str: str) -> tuple[int, int, int] | tuple[None, None, None]:
    '''
    Parses White and Black Elo strings, returning integers and rating difference or None if invalid.
    '''
    try:
        w_elo = int(w_elo_str)
        b_elo = int(b_elo_str)
        rating_diff: int = w_elo - b_elo
        return w_elo, b_elo, rating_diff
    except ValueError:
        return (None, None, None)
    
def parse_result(result_str: str) -> int | None:
    '''
    Parses the Result string to determine winner bit (1 for White win, 0 for Black win, None for draw/unknown).
    '''
    if result_str == '1-0':
        return 1
    elif result_str == '0-1':
        return 0
    else:
        return None
    
def parse_upset(rating_diff: int, winner_bit: int) -> int:
    '''
    Determines if the game was an upset based on rating difference and winner.
    '''
    return 1 if (rating_diff > 0 and winner_bit == 0) or (rating_diff < 0 and winner_bit == 1) else 0

def parse_moves(mainline_moves: Mainline[Move], max_plies: int | None = 24) -> tuple[str, str, int, int, int, int, int, int]:
    '''
    Parses the first `max_plies` plies into white and black move strings, 
    along with castling, development piece counts, and early queen activity.
    '''
    move_seq: list[str] = [str(move) for move in mainline_moves]
    opening_moves: list[str] = move_seq[:max_plies] if max_plies is not None else move_seq
    
    w_moves_list = opening_moves[::2]
    b_moves_list = opening_moves[1::2]
    
    white_moves: str = ' '.join(w_moves_list)
    black_moves: str = ' '.join(b_moves_list)
    white_castled: int = 1 if ('e1g1' in white_moves or 'e1c1' in white_moves) else 0
    black_castled: int = 1 if ('e8g8' in black_moves or 'e8c8' in black_moves) else 0
    white_developed: int = len(set(m[:2] for m in w_moves_list))
    black_developed: int = len(set(m[:2] for m in b_moves_list))
    white_queen_moved: int = 1 if any(m[:2] == 'd1' for m in w_moves_list) else 0
    black_queen_moved: int = 1 if any(m[:2] == 'd8' for m in b_moves_list) else 0
    
    return white_moves, black_moves, white_castled, black_castled, white_developed, black_developed, white_queen_moved, black_queen_moved

def parse_termination(termination_str: str, terminations: list[str] = ['Normal', 'Time forfeit']) -> Literal['Normal', 'Time forfeit'] | None:
    '''
    Parses the Termination string to categorize how the game ended.
    '''
    for term in terminations:
        if term in termination_str:
            return term
    return None

def parse_skill_level(w_elo: int, b_elo: int) -> tuple[int, Literal['Beginner', 'Intermediate', 'Advanced', 'Master']]:
    '''
    Categorizes skill level based on average game Elo.
    '''
    game_elo: float = (w_elo + b_elo) / 2
    if game_elo < 1200:
        return game_elo, 'Beginner'
    elif game_elo < 1600:
        return game_elo, 'Intermediate'
    elif game_elo < 2000:
        return game_elo, 'Advanced'
    else:
        return game_elo, 'Master'