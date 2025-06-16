import React, { useEffect, useRef, useState } from 'react';
import { Activity, Clock, Target } from 'lucide-react';

interface GameState {
  snake: [number, number][];
  food: [number, number];
  score: number;
  game_over: boolean;
  direction?: string;
}

interface GameVisualizationProps {
  gameState: GameState | null;
  isRunning: boolean;
  algorithm: string;
  setGameState: any
}

const GameVisualization: React.FC<GameVisualizationProps> = ({ 
  gameState, 
  isRunning, 
  algorithm,
  setGameState
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [gameSpeed, setGameSpeed] = useState(200);

  const GRID_SIZE = 20;
  const CANVAS_WIDTH = 800;
  const CANVAS_HEIGHT = 600;

  useEffect(() => {
    if (!isRunning || !gameState) return;

    const gameLoop = setInterval(async () => {
      try {
        const response = await fetch('http://localhost:5000/api/game/step', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        if (data.success) {
          setGameState(data.game_state);
          if (data.metrics) {
            setMetrics(data.metrics);
          }
        } else {
          console.error('Game step failed');
        }
      } catch (error) {
        console.error('Game step failed:', error);
      }
    }, gameSpeed);

    return () => clearInterval(gameLoop);
  }, [isRunning, gameSpeed]);

  useEffect(() => {
    if (!gameState) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const renderFrame = () => {
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 1;
      for (let x = 0; x <= CANVAS_WIDTH; x += GRID_SIZE) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, CANVAS_HEIGHT);
        ctx.stroke();
      }
      for (let y = 0; y <= CANVAS_HEIGHT; y += GRID_SIZE) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(CANVAS_WIDTH, y);
        ctx.stroke();
      }

      gameState.snake.forEach((segment, index) => {
        const [x, y] = segment;
        const pixelX = x * GRID_SIZE;
        const pixelY = y * GRID_SIZE;

        if (index === 0) {
          ctx.fillStyle = '#10b981';
          ctx.fillRect(pixelX + 2, pixelY + 2, GRID_SIZE - 4, GRID_SIZE - 4);
          
          ctx.fillStyle = '#ffffff';
          ctx.fillRect(pixelX + 5, pixelY + 5, 3, 3);
          ctx.fillRect(pixelX + 12, pixelY + 5, 3, 3);
        } else {
          const alpha = Math.max(0.3, 1 - (index * 0.1));
          ctx.fillStyle = `rgba(34, 197, 94, ${alpha})`;
          ctx.fillRect(pixelX + 1, pixelY + 1, GRID_SIZE - 2, GRID_SIZE - 2);
        }
      });

      const [foodX, foodY] = gameState.food;
      const foodPixelX = foodX * GRID_SIZE;
      const foodPixelY = foodY * GRID_SIZE;
      
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(
        foodPixelX + GRID_SIZE / 2,
        foodPixelY + GRID_SIZE / 2,
        GRID_SIZE / 2 - 2,
        0,
        2 * Math.PI
      );
      ctx.fill();

      if (gameState.game_over) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
        
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 48px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Game Over', CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2);
        
        ctx.font = '24px Arial';
        ctx.fillText(`Final Score: ${gameState.score}`, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 + 50);
      }
    };

    renderFrame();
  }, [gameState]);

  return (
    <>
    <div className="space-y-6">
      {/* Game Canvas */}
      <div className="flex justify-center">
        <div className="relative">
          <canvas
            ref={canvasRef}
            width={CANVAS_WIDTH}
            height={CANVAS_HEIGHT}
            className="border-2 border-slate-600 rounded-lg bg-slate-900"
          />
          
          {!gameState && (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-900 rounded-lg">
              <div className="text-center text-slate-400">
                <Target size={48} className="mx-auto mb-4" />
                <p className="text-lg">Select an algorithm and start the game</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Game Controls */}
      <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
        <div className="flex items-center gap-4">
          <label className="text-white font-medium">Game Speed:</label>
          <input
            type="range"
            min="50"
            max="500"
            value={gameSpeed}
            onChange={(e) => setGameSpeed(Number(e.target.value))}
            className="w-32"
            disabled={!isRunning}
          />
          <span className="text-slate-300 text-sm">{gameSpeed}ms</span>
        </div>
        
        <div className="flex items-center gap-2 text-slate-300">
          <Activity size={16} />
          <span className="capitalize">{algorithm.replace('_', ' ')} Algorithm</span>
        </div>
      </div>

      {/* Game Metrics */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-slate-700/50 rounded-lg p-4">
            <div className="flex items-center gap-2 text-blue-400 mb-2">
              <Clock size={16} />
              <span className="font-medium">Decision Time</span>
            </div>
            <p className="text-white text-xl font-bold">
              {(metrics.decision_time * 1000).toFixed(2)}ms
            </p>
          </div>
          
          <div className="bg-slate-700/50 rounded-lg p-4">
            <div className="flex items-center gap-2 text-green-400 mb-2">
              <Target size={16} />
              <span className="font-medium">Path Efficiency</span>
            </div>
            <p className="text-white text-xl font-bold">
              {(metrics.path_efficiency * 100).toFixed(1)}%
            </p>
          </div>
          
          <div className="bg-slate-700/50 rounded-lg p-4">
            <div className="flex items-center gap-2 text-purple-400 mb-2">
              <Activity size={16} />
              <span className="font-medium">Moves</span>
            </div>
            <p className="text-white text-xl font-bold">
              {metrics.moves || 0}
            </p>
          </div>
        </div>
      )}
    </div>
    </>
  );
};

export default GameVisualization;