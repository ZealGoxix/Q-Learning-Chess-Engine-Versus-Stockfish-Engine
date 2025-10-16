// Global variables
let board = null;
let progressChart = null;
let game = new Chess();

// Initialize chess board with better error handling
function initBoard() {
    console.log("🔄 Initializing chess board...");
    
    try {
        // Clear any existing board
        const boardElement = document.getElementById('board');
        if (boardElement) {
            boardElement.innerHTML = '';
        }
        
        // Initialize new board
        board = Chessboard('board', {
            draggable: false,
            position: 'start',
            pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
            onLoad: function() {
                console.log("✅ Chess board loaded successfully!");
            }
        });
        
        if (!board) {
            throw new Error("Chessboard initialization failed");
        }
        
        console.log("🎯 Board initialized:", board);
        
    } catch (error) {
        console.error("❌ Error initializing board:", error);
        // Fallback: create a simple visual representation
        const boardElement = document.getElementById('board');
        if (boardElement) {
            boardElement.innerHTML = `
                <div style="width: 400px; height: 400px; background: #f0d9b5; 
                            display: grid; grid-template-columns: repeat(8, 1fr); 
                            grid-template-rows: repeat(8, 1fr); border: 2px solid #333;">
                    ${Array.from({length: 64}, (_, i) => {
                        const row = Math.floor(i / 8);
                        const col = i % 8;
                        const isLight = (row + col) % 2 === 0;
                        return `<div style="background: ${isLight ? '#f0d9b5' : '#b58863'};"></div>`;
                    }).join('')}
                </div>
                <div style="text-align: center; margin-top: 10px; color: red;">
                    Chess board failed to load - but training will still work!
                </div>
            `;
        }
    }
}

// Initialize progress chart
function initChart() {
    try {
        const ctx = document.getElementById('progressChart').getContext('2d');
        progressChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Q-Learning Win Rate (%)',
                    data: [],
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
        console.log("✅ Progress chart initialized");
    } catch (error) {
        console.error("❌ Error initializing chart:", error);
    }
}

// Update board position
function updateBoard(fen) {
    if (board && typeof board.position === 'function') {
        try {
            board.position(fen);
        } catch (error) {
            console.error("Error updating board position:", error);
        }
    }
}

// Start training session
async function startTraining() {
    const numGames = document.getElementById('numGames').value;
    
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
    document.getElementById('trainingStatus').textContent = 'Starting training session...';
    document.getElementById('trainingStatus').className = 'status-indicator status-active';
    
    console.log("🎮 Starting training with", numGames, "games");
    
    try {
        const response = await fetch('/api/start_training', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({num_games: parseInt(numGames)})
        });
        
        const data = await response.json();
        console.log("📡 Start training response:", data);
        
        if (data.error) {
            alert('Error: ' + data.error);
            resetTrainingButtons();
        } else {
            document.getElementById('trainingStatus').textContent = data.message;
            if (!data.stockfish_available) {
                document.getElementById('stockfishWarning').style.display = 'block';
                document.getElementById('blackPlayer').textContent = 'Random (Black)';
            }
        }
        
    } catch (error) {
        console.error("❌ Error starting training:", error);
        alert('Error starting training: ' + error.message);
        resetTrainingButtons();
    }
}

// Stop training session
async function stopTraining() {
    try {
        await fetch('/api/stop_training');
        document.getElementById('trainingStatus').textContent = 'Training stopped by user';
        document.getElementById('trainingStatus').className = 'status-indicator status-inactive';
        resetTrainingButtons();
    } catch (error) {
        console.error("❌ Error stopping training:", error);
        alert('Error stopping training: ' + error.message);
    }
}

// Reset training buttons to default state
function resetTrainingButtons() {
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
}

// Update display with current training status
async function updateDisplay() {
    try {
        const response = await fetch('/api/training_status');
        const data = await response.json();
        
        // Update training status
        const statusEl = document.getElementById('trainingStatus');
        if (data.training_active) {
            statusEl.textContent = data.live_game.status;
            statusEl.className = 'status-indicator status-active';
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
        } else {
            if (statusEl.textContent.includes('Starting') || statusEl.textContent.includes('In progress')) {
                statusEl.textContent = 'Training completed';
            }
            statusEl.className = 'status-indicator status-inactive';
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }
        
        // Update chess board
        if (data.live_game.current_fen && data.live_game.current_fen !== 'start') {
            updateBoard(data.live_game.current_fen);
        }
        
        // Update moves log
        const movesLog = document.getElementById('movesLog');
        if (data.live_game.moves && data.live_game.moves.length > 0) {
            movesLog.innerHTML = data.live_game.moves.map(move => 
                `<div class="move-entry">${move}</div>`
            ).join('');
            movesLog.scrollTop = movesLog.scrollHeight;
        } else {
            movesLog.innerHTML = 'No moves yet...';
        }
        
        // Update current game
        document.getElementById('currentGame').textContent = data.live_game.game_number || '-';
        
        // Update metrics
        if (data.metrics) {
            document.getElementById('totalGames').textContent = data.metrics.total_games;
            document.getElementById('qWins').textContent = data.metrics.q_learning_wins;
            document.getElementById('sWins').textContent = data.metrics.stockfish_wins;
            document.getElementById('draws').textContent = data.metrics.draws;
            document.getElementById('winRate').textContent = data.metrics.win_rate + '%';
            document.getElementById('avgMoves').textContent = data.metrics.avg_moves;
            document.getElementById('explorationRate').textContent = data.metrics.exploration_rate;
            document.getElementById('statesLearned').textContent = data.metrics.states_learned;
        }
        
        // Update game history
        const historyEl = document.getElementById('gameHistory');
        if (data.recent_games && data.recent_games.length > 0) {
            historyEl.innerHTML = data.recent_games.map(game => `
                <div class="game-item winner-${game.winner}">
                    <span>Game ${game.game_number}</span>
                    <span>${game.winner === 'q_learning' ? 'Q-Learning Won' : 
                            game.winner === 'stockfish' ? 'Stockfish Won' : 'Draw'}</span>
                    <span>${game.moves} moves</span>
                    <span>Exploration: ${game.exploration_rate}</span>
                </div>
            `).join('');
        } else {
            historyEl.innerHTML = '<div class="game-item">No games played yet...</div>';
        }
        
        // Show/hide Stockfish warning
        if (!data.stockfish_available) {
            document.getElementById('stockfishWarning').style.display = 'block';
            document.getElementById('blackPlayer').textContent = 'Random (Black)';
        }
        
    } catch (error) {
        console.error('❌ Error updating display:', error);
    }
}

// Update progress chart
async function updateProgressChart() {
    try {
        const response = await fetch('/api/agent_progress');
        const data = await response.json();
        
        if (progressChart && data.progress && data.progress.length > 0) {
            const labels = data.progress.map(p => `Game ${p.game_number}`);
            const winRates = [];
            
            // Calculate cumulative win rate
            let qWins = 0;
            data.progress.forEach((game, index) => {
                if (game.winner === 'q_learning') qWins++;
                winRates.push((qWins / (index + 1)) * 100);
            });
            
            progressChart.data.labels = labels;
            progressChart.data.datasets[0].data = winRates;
            progressChart.update();
        }
        
    } catch (error) {
        console.error('❌ Error updating chart:', error);
    }
}

// Initialize on page load
window.addEventListener('load', function() {
    console.log("🚀 Page loaded, initializing application...");
    
    // Initialize components
    initBoard();
    initChart();
    
    // Initial display update
    updateDisplay();
    
    // Set up periodic updates
    setInterval(updateDisplay, 1000);
    setInterval(updateProgressChart, 5000);
    
    console.log("✅ Application initialized successfully!");
});

// Handle page visibility changes
document.addEventListener('visibilitychange', function() {
    if (!document.hidden) {
        updateDisplay();
    }
});