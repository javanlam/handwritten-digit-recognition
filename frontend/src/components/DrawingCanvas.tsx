import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GridData } from '../types';
import './DrawingCanvas.css';

interface DrawingCanvasProps {
  rows?: number;
  cols?: number;
  cellSize?: number;
  brushSize?: number;
  onGridChange?: (grid: number[][]) => void;
}

const DrawingCanvas: React.FC<DrawingCanvasProps> = ({
  rows = 28,
  cols = 28,
  cellSize = 20,
  brushSize = 2,
  onGridChange,
}) => {
  const [grid, setGrid] = useState<number[][]>(
    Array(rows).fill(0).map(() => Array(cols).fill(0))
  );
  const [isDrawing, setIsDrawing] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas dimensions
    canvas.width = cols * cellSize;
    canvas.height = rows * cellSize;

    // Draw initial grid
    drawGrid(ctx);
  }, [rows, cols, cellSize]);

  // Draw grid lines
  const drawGrid = useCallback((ctx: CanvasRenderingContext2D) => {
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;

    // Draw vertical lines
    for (let col = 0; col <= cols; col++) {
      ctx.beginPath();
      ctx.moveTo(col * cellSize, 0);
      ctx.lineTo(col * cellSize, rows * cellSize);
      ctx.stroke();
    }

    // Draw horizontal lines
    for (let row = 0; row <= rows; row++) {
      ctx.beginPath();
      ctx.moveTo(0, row * cellSize);
      ctx.lineTo(cols * cellSize, row * cellSize);
      ctx.stroke();
    }
  }, [cols, rows, cellSize]);

  // Draw pixel on grid
  const drawPixel = useCallback((row: number, col: number, value: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Calculate pixel position
    const x = col * cellSize;
    const y = row * cellSize;

    // Set fill color based on value (0-255)
    const colorValue = Math.min(255, Math.max(0, value));
    ctx.fillStyle = `rgb(${255 - colorValue}, ${255 - colorValue}, ${255 - colorValue})`;
    
    // Fill the cell
    ctx.fillRect(x, y, cellSize, cellSize);

    // Redraw grid lines
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, cellSize, cellSize);
  }, [cellSize]);

  // Update grid and canvas
  const updateGrid = useCallback((row: number, col: number, value: number) => {
    setGrid(prev => {
      const newGrid = [...prev.map(row => [...row])];
      
      // Apply brush size
      const halfBrush = Math.floor(brushSize / 2);
      for (let dr = -halfBrush; dr <= halfBrush; dr++) {
        for (let dc = -halfBrush; dc <= halfBrush; dc++) {
          const newRow = row + dr;
          const newCol = col + dc;
          
          if (
            newRow >= 0 && newRow < rows &&
            newCol >= 0 && newCol < cols
          ) {
            // Add value with distance-based falloff
            const distance = Math.sqrt(dr * dr + dc * dc);
            const falloff = Math.max(0, 1 - distance / (halfBrush + 1));
            const newValue = Math.min(255, value * falloff);
            
            // Blend with existing value
            newGrid[newRow][newCol] = Math.max(newGrid[newRow][newCol], newValue);
            drawPixel(newRow, newCol, newGrid[newRow][newCol]);
          }
        }
      }
      
      // Notify parent
      if (onGridChange) {
        onGridChange(newGrid);
      }
      
      return newGrid;
    });
  }, [rows, cols, brushSize, drawPixel, onGridChange]);

  // Get canvas coordinates from mouse event
  const getCanvasCoordinates = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const col = Math.floor(x / cellSize);
    const row = Math.floor(y / cellSize);

    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      return { row, col };
    }
    return null;
  };

  // Handle mouse down
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true);
    const coords = getCanvasCoordinates(e);
    if (coords) {
      updateGrid(coords.row, coords.col, 255);
    }
  };

  // Handle mouse move
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    
    const coords = getCanvasCoordinates(e);
    if (coords) {
      updateGrid(coords.row, coords.col, 255);
    }
  };

  // Handle mouse up
  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  // Clear canvas
  const clearCanvas = () => {
    setGrid(Array(rows).fill(0).map(() => Array(cols).fill(0)));
    
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Redraw grid
    drawGrid(ctx);
    
    // Notify parent
    if (onGridChange) {
      onGridChange(Array(rows).fill(0).map(() => Array(cols).fill(0)));
    }
  };

  // Handle touch events
  const handleTouchStart = (e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    setIsDrawing(true);
    const touch = e.touches[0];
    const mockEvent = {
      clientX: touch.clientX,
      clientY: touch.clientY,
    } as React.MouseEvent<HTMLCanvasElement>;
    handleMouseDown(mockEvent);
  };

  const handleTouchMove = (e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const touch = e.touches[0];
    const mockEvent = {
      clientX: touch.clientX,
      clientY: touch.clientY,
    } as React.MouseEvent<HTMLCanvasElement>;
    handleMouseMove(mockEvent);
  };

  const handleTouchEnd = () => {
    setIsDrawing(false);
  };

  return (
    <div className="drawing-container" ref={containerRef}>
      <div className="canvas-wrapper">
        <canvas
          ref={canvasRef}
          className="drawing-canvas"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
        />
      </div>
      <button 
        className="clear-button"
        onClick={clearCanvas}
      >
        Clear Canvas
      </button>
      <div className="instructions">
        <p>Draw a digit (0-9) in the grid above.</p>
        <p>Use mouse or touch to draw.</p>
      </div>
    </div>
  );
};

export default DrawingCanvas;