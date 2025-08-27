import React, { useState } from 'react';
import './MethodSelector.css';

const MethodSelector = ({ 
  availableMethods = [], 
  onMethodsSelected, 
  onOptimizationStart,
  onOptimizationSuccess,
  onOptimizationError,
  isProcessing = false 
}) => {
  const [selectedMethods, setSelectedMethods] = useState([]);
  const [showDescriptions, setShowDescriptions] = useState(false);

  // Method descriptions for user guidance
  const methodDescriptions = {
    "Matrix Algebra": {
      description: "Traditional mean-variance optimization using matrix algebra. Fast and reliable for well-behaved data.",
      pros: ["Quick computation", "Mathematically elegant", "Industry standard"],
      cons: ["Assumes normal distributions", "Sensitive to estimation errors"]
    },
    "CVXPY": {
      description: "Convex optimization approach with flexible constraints. Allows for advanced portfolio constraints.",
      pros: ["Flexible constraints", "Robust optimization", "Handle complex objectives"],
      cons: ["Slower computation", "Requires more memory"]
    },
    "Skewed-t Distribution": {
      description: "Handles non-normal return distributions with skewness and heavy tails. Better for real market data.",
      pros: ["Realistic return assumptions", "Captures market anomalies", "Robust to outliers"],
      cons: ["More complex", "Longer computation time"]
    }
  };

  const handleMethodToggle = (method) => {
    const updatedMethods = selectedMethods.includes(method)
      ? selectedMethods.filter(m => m !== method)
      : [...selectedMethods, method];
    
    setSelectedMethods(updatedMethods);
    onMethodsSelected(updatedMethods);
  };

  const handleOptimize = async () => {
    if (selectedMethods.length === 0) {
      onOptimizationError("Please select at least one optimization method.");
      return;
    }

    try {
      onOptimizationStart();
      
      const response = await fetch('/api/optimize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          methods: selectedMethods
        })
      });

      if (!response.ok) {
        throw new Error(`Optimization failed: ${response.statusText}`);
      }

      const result = await response.json();
      
      if (result.success) {
        onOptimizationSuccess(result);
      } else {
        throw new Error(result.error || 'Optimization failed');
      }
    } catch (error) {
      onOptimizationError(error.message);
    }
  };

  return (
    <div className="method-selector">
      <div className="method-selector-header">
        <h2>Select Optimization Methods</h2>
        <p>Choose one or more methods to analyze your portfolio data</p>
        <button 
          className="toggle-descriptions-btn"
          onClick={() => setShowDescriptions(!showDescriptions)}
        >
          {showDescriptions ? 'Hide' : 'Show'} Method Details
        </button>
      </div>

      <div className="methods-grid">
        {availableMethods.map((method, index) => (
          <div key={method} className="method-card">
            <div className="method-card-header">
              <label className="method-checkbox">
                <input
                  type="checkbox"
                  checked={selectedMethods.includes(method)}
                  onChange={() => handleMethodToggle(method)}
                  disabled={isProcessing}
                />
                <span className="checkmark"></span>
                <span className="method-name">{method}</span>
              </label>
            </div>
            
            {showDescriptions && methodDescriptions[method] && (
              <div className="method-description">
                <p className="description-text">
                  {methodDescriptions[method].description}
                </p>
                <div className="pros-cons">
                  <div className="pros">
                    <h4>Advantages:</h4>
                    <ul>
                      {methodDescriptions[method].pros.map((pro, i) => (
                        <li key={i}>{pro}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="cons">
                    <h4>Considerations:</h4>
                    <ul>
                      {methodDescriptions[method].cons.map((con, i) => (
                        <li key={i}>{con}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="method-selector-actions">
        <div className="selection-summary">
          {selectedMethods.length > 0 ? (
            <span className="selected-count">
              {selectedMethods.length} method{selectedMethods.length !== 1 ? 's' : ''} selected: {selectedMethods.join(', ')}
            </span>
          ) : (
            <span className="no-selection">No methods selected</span>
          )}
        </div>
        
        <button 
          className="optimize-btn"
          onClick={handleOptimize}
          disabled={isProcessing || selectedMethods.length === 0}
        >
          {isProcessing ? (
            <>
              <div className="spinner"></div>
              Optimizing...
            </>
          ) : (
            'Run Optimization'
          )}
        </button>
      </div>
    </div>
  );
};

export default MethodSelector;