import chess
import chess.engine
import numpy as np
import random
import time

class ChessQLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        self.q_table = {}  # Stores Q-values for state-action pairs
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
    def board_to_state(self, board):
        """Convert chess board to simplified state representation"""
        # Use FEN string of the position as state identifier
        return board.fen().split(' ')[0]  # Only position info, not move counts
        
    def get_best_move(self, board):
        """Choose move using epsilon-greedy policy"""
        state = self.board_to_state(board)
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None
            
        # Explore: random move
        if random.random() < self.exploration_rate:
            return random.choice(legal_moves)
            
        # Exploit: best known move
        if state in self.q_table:
            # Find move with highest Q-value
            best_move = None
            best_value = -float('inf')
            
            for move in legal_moves:
                move_str = str(move)
                if move_str in self.q_table[state] and self.q_table[state][move_str] > best_value:
                    best_value = self.q_table[state][move_str]
                    best_move = move
            
            if best_move:
                return best_move
        
        # Fallback: random move
        return random.choice(legal_moves)
    
    def update_q_value(self, board, move, reward, next_board):
        """Update Q-value using Q-learning formula"""
        state = self.board_to_state(board)
        next_state = self.board_to_state(next_board)
        move_str = str(move)
        
        # Initialize state if not exists
        if state not in self.q_table:
            self.q_table[state] = {}
        
        # Get current Q-value
        current_q = self.q_table[state].get(move_str, 0)
        
        # Get maximum Q-value for next state
        next_max = 0
        if next_state in self.q_table and self.q_table[next_state]:
            next_max = max(self.q_table[next_state].values())
        
        # Q-learning formula
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max - current_q
        )
        
        self.q_table[state][move_str] = new_q

def get_reward(board, move, next_board):
    """Calculate reward for a move"""
    reward = 0
    
    # Check for game-ending conditions
    if next_board.is_checkmate():
        # If our move caused checkmate, big reward
        reward = 100
    elif next_board.is_check():
        # Small reward for putting opponent in check
        reward = 5
    elif next_board.is_capture(move):
        # Reward for captures based on piece value
        piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9}
        captured_piece = board.piece_at(move.to_square)
        if captured_piece:
            piece_symbol = captured_piece.symbol().lower()
            reward = piece_values.get(piece_symbol, 1)
    
    return reward

def run_game(q_agent, num_games=1, return_moves=False):
    """Run games between Q-learning agent and Stockfish"""
    results = []
    
    try:
        # Initialize Stockfish engine
        with chess.engine.SimpleEngine.popen_uci("stockfish") as engine:
            
            for game_num in range(num_games):
                board = chess.Board()
                moves = []
                game_log = []
                
                # Q-learning plays as White
                while not board.is_game_over() and len(moves) < 100:  # Limit moves per game
                    
                    if board.turn:  # Q-learning agent's turn (White)
                        current_board = board.copy()
                        move = q_agent.get_best_move(board)
                        
                        if move is None:
                            break
                            
                        board.push(move)
                        reward = get_reward(current_board, move, board)
                        
                        # Update Q-value based on what happens next
                        if not board.is_game_over():
                            next_board = board.copy()
                            # Stockfish responds
                            stockfish_move = engine.play(next_board, chess.engine.Limit(time=0.1)).move
                            next_board.push(stockfish_move)
                            q_agent.update_q_value(current_board, move, reward, next_board)
                        else:
                            q_agent.update_q_value(current_board, move, reward, board)
                    
                    else:  # Stockfish's turn (Black)
                        stockfish_move = engine.play(board, chess.engine.Limit(time=0.1)).move
                        board.push(stockfish_move)
                    
                    moves.append(str(board.peek()) if board.move_stack else "start")
                    game_log.append(board.fen())
                
                # Determine game result
                if board.is_checkmate():
                    winner = "stockfish" if board.turn else "q_learning"
                elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
                    winner = "draw"
                else:
                    winner = "draw"  # Move limit reached
                
                game_result = {
                    'game_number': game_num + 1,
                    'winner': winner,
                    'moves': len(moves),
                    'final_fen': board.fen()
                }
                
                if return_moves:
                    game_result['move_sequence'] = moves
                    game_result['game_log'] = game_log
                
                results.append(game_result)
                
                # Gradually decrease exploration
                q_agent.exploration_rate = max(0.01, q_agent.exploration_rate * 0.995)
    
    except Exception as e:
        print(f"Error in game: {e}")
        # Fallback: random mover if Stockfish not available
        if not results:
            results.append({'game_number': 1, 'winner': 'error', 'moves': 0, 'error': str(e)})
    
    return results if num_games > 1 else results[0] if results else None