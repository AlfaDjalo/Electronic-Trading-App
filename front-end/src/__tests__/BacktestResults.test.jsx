import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom/extend-expect';
import BacktestResults from '../components/BacktestResults';

const mockResults = {
  "bt1": {
    model_name: 'Model A',
    equity: [
      { date: '2020-01-01', value: 100 },
      { date: '2020-01-02', value: 102 }
    ],
    trades: [
      { date: '2020-01-01', price: 100, 'buy/sell': 'buy' },
      { date: '2020-01-02', price: 102, 'buy/sell': 'sell' }
    ],
    stats: { total_return: 2, max_drawdown: -1, sharpe: 0.5, win_rate: 1 }
  }
};

describe('BacktestResults component', () => {
  test('renders stats table and controls', () => {
    render(<BacktestResults backtestResults={mockResults} />);

    // Stats table headings
    expect(screen.getByText(/Backtest Statistics/i)).toBeInTheDocument();
    expect(screen.getByText(/Model/i)).toBeInTheDocument();

    // Model name present
    expect(screen.getByText('Model A')).toBeInTheDocument();

    // Toggle exists and is checked by default
    const checkbox = screen.getByRole('checkbox', { name: /Show trade markers/i });
    expect(checkbox).toBeInTheDocument();
    expect(checkbox).toBeChecked();
  });

  test('toggle hides/shows markers (checkbox behavior)', () => {
    render(<BacktestResults backtestResults={mockResults} />);
    const checkbox = screen.getByRole('checkbox', { name: /Show trade markers/i });
    expect(checkbox).toBeChecked();
    fireEvent.click(checkbox);
    expect(checkbox).not.toBeChecked();
  });
});
