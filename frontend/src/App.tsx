import React, { useState, useCallback } from 'react';
import DrawingCanvas from './components/DrawingCanvas';
import PredictionResultComponent from './components/PredictionResult';
import { InferenceAPI } from './services/api';
import { PredictionResult } from './types';
import './App.css';

const App: React.FC = () => {
  const [grid, setGrid] = useState<number[][]>(
    Array(28).fill(0).map(() => Array(28).fill(0))
  );
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGridChange = useCallback((newGrid: number[][]) => {
    setGrid(newGrid);
  }, []);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await InferenceAPI.predict(grid);
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Display grid values for debugging (optional)
  const displayGridValues = () => {
    console.log('Current grid values (28x28):');
    console.log(grid.map(row => row.map(val => val.toString().padStart(3, ' ')).join(' ')));
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Handwritten Digit Recognition</h1>
        <p>Draw a digit (0-9) in the center of the grid below and click Predict!</p>
      </header>

      <main className="app-main">
        <div className="left-panel">
          <div className="drawing-section">
            <h2>Drawing Canvas</h2>
            <DrawingCanvas
              rows={28}
              cols={28}
              cellSize={20}
              brushSize={3}
              onGridChange={handleGridChange}
            />
          </div>
          
          <div className="controls">
            <button 
              className="predict-button"
              onClick={handlePredict}
              disabled={loading}
            >
              {loading ? 'Predicting...' : 'Predict Digit'}
            </button>
            
            {/* <button 
              className="debug-button"
              onClick={displayGridValues}
              title="Log grid values to console"
            >
              Debug Grid
            </button> */}
          </div>
        </div>

        <div className="right-panel">
          <PredictionResultComponent
            result={prediction}
            loading={loading}
            error={error}
          />
          
          <div className="info-section">
            <h3>About This Model</h3>
            <ul>
              <li>Trained on MNIST dataset</li>
              <li>Uses CNN architecture</li>
              <li><a href="https://github.com/javanlam/handwritten-digit-recognition">Source Code</a></li>
            </ul>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>Handwritten Digit Recognition System</p>
        <p>Draw a digit in the grid and let the model predict what it is!</p>
      </footer>
    </div>
  );
};

export default App;