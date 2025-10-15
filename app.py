from flask import Flask, render_template, jsonify, request
import chess
import chess.engine
from chess_ai import ChessQLearningAgent, run_game
import threading
import os

app = Flask(__name__)

# Global variables to track game state
current_game = None
game_history = []
q_agent = None

@app.route('/')
def index():
    """Main page with chess board and controls"""
    return render_template('index.html')

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """Start training session between Q-learning and Stockfish"""
    global current_game, q_agent
    
    data = request.json
    num_games = data.get('num_games', 10)
    
    # Initialize Q-learning agent if first time
    if q_agent is None:
        q_agent = ChessQLearningAgent()
    
    # Run games in background thread
    def train_background():
        global game_history
        results = run_game(q_agent, num_games=num_games)
        game_history.extend(results)
    
    thread = threading.Thread(target=train_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started', 'games': num_games})

@app.route('/api/game_history')
def get_game_history():
    """Get history of all games played"""
    return jsonify({'games': game_history[-20:]})  # Last 20 games

@app.route('/api/agent_stats')
def get_agent_stats():
    """Get Q-learning agent performance statistics"""
    if q_agent is None:
        return jsonify({'error': 'Agent not trained yet'})
    
    # Calculate basic stats from game history
    if game_history:
        recent_games = game_history[-50:]  # Last 50 games
        wins = sum(1 for game in recent_games if game['winner'] == 'q_learning')
        losses = sum(1 for game in recent_games if game['winner'] == 'stockfish')
        draws = sum(1 for game in recent_games if game['winner'] == 'draw')
        
        stats = {
            'total_games': len(game_history),
            'recent_wins': wins,
            'recent_losses': losses, 
            'recent_draws': draws,
            'win_rate': (wins / len(recent_games)) * 100 if recent_games else 0,
            'states_learned': len(q_agent.q_table)
        }
    else:
        stats = {'total_games': 0, 'win_rate': 0, 'states_learned': 0}
    
    return jsonify(stats)

@app.route('/api/watch_game')
def watch_live_game():
    """Run a single game and return move-by-move data"""
    if q_agent is None:
        q_agent = ChessQLearningAgent()
    
    # Run one game and return all moves
    game_result = run_game(q_agent, num_games=1, return_moves=True)
    
    return jsonify({'live_game': game_result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)