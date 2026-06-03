from pathlib import Path
import pandas as pd
import io
import os
from chess.pgn import Game, read_game
import zstandard as zstd

def load_and_process_lichess_data(zst_path: Path, output_csv_path: Path, target_rows: int = 100000):
    '''
    Streams raw .pgn.zst [Lichess](https://database.lichess.org/#standard_games) data, extracts key features for the predictive task,
    and saves a clean tabular snapshot to CSV.
    '''
    
    if os.path.exists(output_csv_path):
        print(f'Clean data already exists at {output_csv_path}. Loading existing file...')
        return pd.read_csv(output_csv_path)
        
    if not os.path.exists(zst_path):
        raise FileNotFoundError(f'Raw source archive not found at {zst_path}. Please download it first.')
    
    games_data: list[dict] = []

    with open(zst_path, 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            
            while len(games_data) < target_rows:
                game: Game = read_game(text_stream)
                if game is None:
                    print('Reached end of file archive.')
                    break
                    
                headers: dict[str, str] = game.headers
                
                white_elo: str = headers.get('WhiteElo', '?')
                black_elo: str = headers.get('BlackElo', '?')
                result: str = headers.get('Result', '?')  
                opening_name: str = headers.get('Opening', '?')
                eco_code: str = headers.get('ECO', '?')   
                time_control: str = headers.get('TimeControl', '?')
                
                if white_elo == '?' or black_elo == '?' or result == '*' or result == '1/2-1/2':
                    continue
                    
                moves_seq: list[str] = [str(move) for move in game.mainline_moves()]
                opening_moves: str = ' '.join(moves_seq[:6])
                
                if not opening_moves:
                    continue

                games_data.append({
                    'white_elo': int(white_elo),
                    'black_elo': int(black_elo),
                    'rating_diff': int(white_elo) - int(black_elo),
                    'opening_eco': eco_code,
                    'opening_name': opening_name,
                    'opening_moves': opening_moves,
                    'winner': 1 if result == '1-0' else 0 
                })

    df = pd.DataFrame(games_data)
    
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f'Successfully compiled and saved {len(df):,} rows to {output_csv_path}!')
    
    return df

if __name__ == '__main__':
    RAW_FILE: str = 'lichess_db_standard_rated_2016-07.pgn.zst'
    RAW_PATH: Path = Path('data') / RAW_FILE
    
    PROCESSED_FILE: str = 'lichess_processed_100k.csv'
    PROCESSED_PATH: Path = Path('data') / PROCESSED_FILE
    
    load_and_process_lichess_data(RAW_PATH, PROCESSED_PATH)