import chess.engine
import subprocess

try:
    # Test if stockfish command works
    result = subprocess.run(["which", "stockfish"], capture_output=True, text=True)
    print("Stockfish path:", result.stdout.strip())
    
    # Test if we can start engine
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    print("✅ Stockfish is working!")
    engine.quit()
except Exception as e:
    print("❌ Stockfish error:", e)