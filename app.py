from flask import Flask, render_template, jsonify, request
import chess
import chess.engine
import numpy as np
import random
import time
import threading
import os
import subprocess

app = Flask(__name__)

# Global training state
training_active = False
training_results = []
live_game_data = {
    'current_fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    'moves': [],
    'game_number': 0,
    'status': 'Ready to start training'
}

class ChessQLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.5 # Initial exploration rate
        
    def get_state_key(self, board):
        """Create a simplified state representation"""
        # Use material count and piece positions as state
        material = self.calculate_material(board)
        return f"{material}_{board.turn}"
    
    def calculate_material(self, board):
        """Calculate material advantage for white"""
        piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values.get(piece.symbol().lower(), 0)
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        return white_material - black_material
    
    def get_move(self, board):
        """Choose move using epsilon-greedy policy"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        state = self.get_state_key(board)
        
        # Initialize state if not seen before
        if state not in self.q_table:
            self.q_table[state] = {str(move): 0 for move in legal_moves}
        
        # Explore: random move
        if random.random() < self.exploration_rate:
            return random.choice(legal_moves)
        
        # Exploit: best known move
        q_values = self.q_table[state]
        best_move = None
        best_value = -float('inf')
        
        for move in legal_moves:
            move_str = str(move)
            value = q_values.get(move_str, 0)
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move if best_move else random.choice(legal_moves)
    
    def update(self, old_board, move, reward, new_board):
        """Update Q-values using Q-learning"""
        old_state = self.get_state_key(old_board)
        new_state = self.get_state_key(new_board)
        move_str = str(move)
        
        # Initialize states if needed
        if old_state not in self.q_table:
            self.q_table[old_state] = {}
        if new_state not in self.q_table:
            self.q_table[new_state] = {}
        
        # Get current Q-value
        current_q = self.q_table[old_state].get(move_str, 0)
        
        # Get maximum future Q-value
        max_future_q = max(self.q_table[new_state].values()) if self.q_table[new_state] else 0
        
        # Q-learning formula
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )
        
        self.q_table[old_state][move_str] = new_q

def calculate_reward(old_board, move, new_board):
    """Calculate reward for a move"""
    reward = 0
    
    # Check for game outcome
    if new_board.is_checkmate():
        if new_board.turn:  # Black just moved, so White (Q-learning) won
            reward = 100
        else:
            reward = -100
    elif new_board.is_stalemate() or new_board.is_insufficient_material():
        reward = 10  # Small reward for draw
    elif new_board.is_check():
        reward = 5  # Reward for check
    elif old_board.is_capture(move):
        # Reward based on captured piece value
        piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9}
        captured_piece = old_board.piece_at(move.to_square)
        if captured_piece:
            piece_symbol = captured_piece.symbol().lower()
            reward = piece_values.get(piece_symbol, 1)
    
    return reward

def get_stockfish_path():
    """Find Stockfish executable path"""
    possible_paths = [
        '/usr/games/stockfish',
        '/usr/bin/stockfish', 
        '/usr/local/bin/stockfish',
        'stockfish'  # Try PATH
    ]
    
    for path in possible_paths:
        try:
            # Test if Stockfish works at this path
            engine = chess.engine.SimpleEngine.popen_uci(path)
            engine.quit()
            print(f"✅ Found Stockfish at: {path}")
            return path
        except:
            continue
    
    print("❌ Stockfish not found in common locations")
    return None

# Global Q-learning agent
q_agent = ChessQLearningAgent()
stockfish_path = get_stockfish_path()

def run_training_session(num_games):
    """Run training session in background"""
    global training_active, live_game_data, training_results
    
    training_active = True
    session_results = []
    
    try:
        # Initialize Stockfish if available
        engine = None
        if stockfish_path:
            try:
                engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                print("✅ Stockfish engine started successfully")
            except Exception as e:
                print(f"❌ Failed to start Stockfish: {e}")
                engine = None
        
        for game_num in range(num_games):
            if not training_active:
                break
                
            board = chess.Board()
            game_moves = []
            q_agent.exploration_rate = max(0.1, 0.5 * (0.99 ** game_num))
            
            # Update live game data
            live_game_data.update({
                'current_fen': board.fen(),
                'moves': [],
                'game_number': game_num + 1,
                'status': f'Game {game_num + 1}/{num_games} - In progress...'
            })
            
            while not board.is_game_over() and len(game_moves) < 50:
                if not training_active:
                    break
                    
                if board.turn:  # Q-learning agent's turn (White)
                    old_board = board.copy()
                    move = q_agent.get_move(board)
                    
                    if not move:
                        break
                        
                    board.push(move)
                    reward = calculate_reward(old_board, move, board)
                    game_moves.append(f"Q-Learning: {move}")
                    
                    # Update live display
                    live_game_data['current_fen'] = board.fen()
                    live_game_data['moves'] = game_moves[-10:]
                    
                    # Update Q-values if game continues
                    if not board.is_game_over():
                        if engine:
                            # Get Stockfish's response for learning
                            stockfish_move = engine.play(board, chess.engine.Limit(time=0.1)).move
                            new_board = board.copy()
                            new_board.push(stockfish_move)
                            q_agent.update(old_board, move, reward, new_board)
                        else:
                            # Fallback: random move for black
                            legal_moves = list(board.legal_moves)
                            if legal_moves:
                                stockfish_move = random.choice(legal_moves)
                                new_board = board.copy()
                                new_board.push(stockfish_move)
                                q_agent.update(old_board, move, reward, new_board)
                    else:
                        q_agent.update(old_board, move, reward, board)
                        
                else:  # Black's turn (Stockfish or random)
                    if engine:
                        move = engine.play(board, chess.engine.Limit(time=0.1)).move
                        game_moves.append(f"Stockfish: {move}")
                    else:
                        # Fallback: random move
                        legal_moves = list(board.legal_moves)
                        if legal_moves:
                            move = random.choice(legal_moves)
                            game_moves.append(f"Random: {move}")
                        else:
                            break
                    
                    board.push(move)
                    
                    # Update live display
                    live_game_data['current_fen'] = board.fen()
                    live_game_data['moves'] = game_moves[-10:]
                
                time.sleep(0.3)  # Slow down for watching
            
            # Determine game result
            if board.is_checkmate():
                winner = "q_learning" if not board.turn else "stockfish"
            else:
                winner = "draw"
            
            game_result = {
                'game_number': game_num + 1,
                'winner': winner,
                'moves': len(game_moves),
                'exploration_rate': round(q_agent.exploration_rate, 3),
                'states_learned': len(q_agent.q_table)
            }
            
            session_results.append(game_result)
            training_results.append(game_result)
            
            # Update final game state
            live_game_data.update({
                'current_fen': board.fen(),
                'status': f'Game {game_num + 1}/{num_games} - Complete! Winner: {winner}'
            })
            
            time.sleep(1)
        
        if engine:
            engine.quit()
        
    except Exception as e:
        print(f"Training error: {e}")
        live_game_data['status'] = f'Error: {str(e)}'
    
    training_active = False
    return session_results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """Start a new training session"""
    global training_active
    
    if training_active:
        return jsonify({'error': 'Training already in progress'})
    
    data = request.json
    num_games = data.get('num_games', 10)
    
    # Start training in background thread
    thread = threading.Thread(target=lambda: run_training_session(num_games))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'started', 
        'num_games': num_games,
        'message': f'Started training session with {num_games} games',
        'stockfish_available': stockfish_path is not None
    })

@app.route('/api/stop_training')
def stop_training():
    """Stop current training session"""
    global training_active
    training_active = False
    return jsonify({'status': 'stopped'})

@app.route('/api/training_status')
def training_status():
    """Get current training status and live game data"""
    global training_active, training_results
    
    # Calculate metrics
    if training_results:
        recent_games = training_results
        q_wins = sum(1 for g in recent_games if g['winner'] == 'q_learning')
        s_wins = sum(1 for g in recent_games if g['winner'] == 'stockfish')
        draws = sum(1 for g in recent_games if g['winner'] == 'draw')
        win_rate = (q_wins / len(recent_games)) * 100 if recent_games else 0
        avg_moves = sum(g['moves'] for g in recent_games) / len(recent_games) if recent_games else 0
    else:
        q_wins = s_wins = draws = win_rate = avg_moves = 0
    
    return jsonify({
        'training_active': training_active,
        'live_game': live_game_data,
        'metrics': {
            'total_games': len(training_results),
            'q_learning_wins': q_wins,
            'stockfish_wins': s_wins,
            'draws': draws,
            'win_rate': round(win_rate, 1),
            'avg_moves': round(avg_moves, 1),
            'exploration_rate': round(q_agent.exploration_rate, 3),
            'states_learned': len(q_agent.q_table)
        },
        'recent_games': training_results[-10:],
        'stockfish_available': stockfish_path is not None
    })

@app.route('/api/agent_progress')
def agent_progress():
    """Get Q-learning agent learning progress"""
    progress_data = []
    for i, game in enumerate(training_results):
        if i % 2 == 0 or i == len(training_results) - 1:  # Sample every 2 games
            progress_data.append({
                'game_number': game['game_number'],
                'winner': game['winner'],
                'states_learned': game['states_learned']
            })
    
    return jsonify({'progress': progress_data})

if __name__ == '__main__':
    print("🚀 Starting Chess AI Training Server...")
    if stockfish_path:
        print("✅ Stockfish is available")
    else:
        print("❌ Stockfish not found - using random moves for black")
    app.run(host='0.0.0.0', port=5000, debug=True)