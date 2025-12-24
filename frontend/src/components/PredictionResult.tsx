import React from 'react';
import { PredictionResult } from '../types';
import './PredictionResult.css';

interface PredictionResultProps {
  result: PredictionResult | null;
  loading: boolean;
  error: string | null;
}

const PredictionResultComponent: React.FC<PredictionResultProps> = ({
  result,
  loading,
  error,
}) => {
  if (loading) {
    return (
      <div className="prediction-container loading">
        <div className="spinner"></div>
        <p>Analyzing your drawing...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="prediction-container error">
        <h3>Error</h3>
        <p>{error}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="prediction-container">
        <h3>Prediction Results</h3>
        <p>Draw a digit and click "Predict" to see results.</p>
      </div>
    );
  }

  // Format probability as percentage
  const formatProbability = (prob: number) => {
    return `${(prob * 100).toFixed(2)}%`;
  };

  return (
    <div className="prediction-container">
      <h3>Prediction Results</h3>
      
      <div className="primary-prediction">
        <div className="prediction-card">
          <div className="predicted-digit">
            {result.predicted_class}
          </div>
          <div className="confidence">
            Confidence: {formatProbability(result.probability)}
          </div>
        </div>
      </div>

      <div className="top-predictions">
        <h4>Top 3 Predictions:</h4>
        <div className="predictions-list">
          {result.top3_classes.map((digit, index) => (
            <div key={digit} className="prediction-item">
              <div className="prediction-rank">#{index + 1}</div>
              <div className="prediction-digit">{digit}</div>
              <div className="prediction-bar">
                <div 
                  className="probability-fill"
                  style={{ 
                    width: `${result.top3_probabilities[index] * 100}%` 
                  }}
                />
              </div>
              <div className="probability-value">
                {formatProbability(result.top3_probabilities[index])}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default PredictionResultComponent;