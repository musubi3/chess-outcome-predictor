from pathlib import Path
import pandas as pd
import io
import os
from chess.pgn import Game, read_game
import zstandard as zstd
from src.features import parse_moves, parse_result, parse_termination, parse_time_control, parse_event, parse_elo, parse_upset, parse_skill_level
from typing import Any

def load_and_process_lichess_data(zst_path: Path, output_csv_path: Path, target_rows: int = 100000):
    '''
    Streams raw [.pgn.zst](https://database.lichess.org/#standard_games) data, extracts environmental & tactical features 
    for the Upset Anomaly Predictive task, and saves a tabular snapshot to CSV.
    '''
    if os.path.exists(output_csv_path):
        print(f'Clean data already exists at {output_csv_path}. Loading existing file...')
        return pd.read_csv(output_csv_path)
        
    if not os.path.exists(zst_path):
        raise FileNotFoundError(f'Raw source archive not found at {zst_path}. Please download it first.')
    
    games_data: list[dict[str, Any]] = []

    print(f'Processing raw stream to extract {target_rows:,} clean chess anomalies...')
    with open(zst_path, 'rb') as fh:
        dctx: zstd.ZstdDecompressor = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream: io.TextIOWrapper = io.TextIOWrapper(reader, encoding='utf-8')
            
            while len(games_data) < target_rows:
                game: Game = read_game(text_stream)
                if game is None:
                    print('Reached end of file archive.')
                    break
                    
                headers: dict[str, str] = game.headers
                feature_strs: list[str] = ['WhiteElo', 'BlackElo', 'Result', 'Opening', 'ECO', 'TimeControl', 'Event', 'Termination']
                features: dict[str, str] = {feature: headers.get(feature, '?') for feature in feature_strs}
                if features['Result'] in ['*', '1/2-1/2']:
                    continue
                
                w_elo_int: int; b_elo_int: int; rating_diff: int
                w_elo_int, b_elo_int, rating_diff = parse_elo(features['WhiteElo'], features['BlackElo'])
                if not w_elo_int or not b_elo_int:
                    continue
                
                game_elo: float; skill_tier: str
                game_elo, skill_tier = parse_skill_level(w_elo_int, b_elo_int)
                
                white_moves: str; black_moves: str; white_castled: int; black_castled: int; white_developed: int; black_developed: int; white_queen_moved: int; black_queen_moved: int
                white_moves, black_moves, white_castled, black_castled, white_developed, black_developed, white_queen_moved, black_queen_moved = parse_moves(game.mainline_moves())
                if len(white_moves.split() + black_moves.split()) < 6:
                    continue
                
                winner_bit: int = parse_result(features['Result'])
                is_upset: int = parse_upset(rating_diff, winner_bit)
                
                base_time: int; increment: int
                base_time, increment = parse_time_control(features['TimeControl']) 
                
                speed_category: str = parse_event(features['Event'])
                if not speed_category:
                    continue
                
                termination_category: str = parse_termination(features['Termination'])
                if not termination_category:
                    continue   

                games_data.append({
                    'skill_tier': skill_tier,
                    'game_elo': game_elo,
                    'white_elo': w_elo_int,
                    'black_elo': b_elo_int,
                    'abs_rating_diff': abs(rating_diff),
                    'higher_rated_color': 1 if rating_diff > 0 else 0,
                    'base_time': base_time,
                    'increment': increment,
                    'speed_category': speed_category,
                    'opening_eco': features['ECO'],
                    'opening_name': features['Opening'],
                    'white_moves': white_moves,
                    'black_moves': black_moves,
                    'white_castled': white_castled,
                    'black_castled': black_castled,
                    'white_developed': white_developed,
                    'black_developed': black_developed,
                    'white_queen_moved': white_queen_moved,
                    'black_queen_moved': black_queen_moved,
                    'is_upset': is_upset,
                    'termination_category': termination_category,
                    'winner': winner_bit
                })

    df: pd.DataFrame = pd.DataFrame(games_data)
    
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f'Successfully compiled and saved {len(df):,} rows to {output_csv_path}!')
    
    return df