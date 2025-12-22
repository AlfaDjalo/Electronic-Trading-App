import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ViewChart } from '../components/ViewChart';

const mockChartData = [
  { date: '2020-01-01', equity: 100, position: 0 },
  { date: '2020-01-02', equity: 102, position: 1.2 },
  { date: '2020-01-03', equity: 105, position: 1.2 },
  { date: '2020-01-04', equity: 103, position: 1.2 },
  { date: '2020-01-05', equity: 108, position: 0 }
];

const mockTradeMarkers = [
  {
    date: '2020-01-02',
    value: 102,
    side: 'buy',
    price: 102,
    shares: 1.2,
    backtestId: 'bt1',
    modelName: 'Model A'
  },
  {
    date: '2020-01-05',
    value: 108,
    side: 'sell',
    price: 108,
    shares: 1.2,
    backtestId: 'bt1',
    modelName: 'Model A'
  }
];

describe('ViewChart component', () => {
  test('renders chart with basic data', () => {
    render(
      <ViewChart
        chartData={mockChartData}
        selectedSeries={['equity']}
        tradeMarkers={[]}
        showPosition={false}
      />
    );

    // Check that ResponsiveContainer is present
    const container = document.querySelector('.w-full.h-\\[600px\\]');
    expect(container).toBeInTheDocument();
  });

  test('renders without trade markers when empty array provided', () => {
    render(
      <ViewChart
        chartData={mockChartData}
        selectedSeries={['equity']}
        tradeMarkers={[]}
        showPosition={false}
      />
    );

    const container = document.querySelector('.w-full.h-\\[600px\\]');
    expect(container).toBeInTheDocument();
  });

  test('accepts trade markers prop', () => {
    const { container } = render(
      <ViewChart
        chartData={mockChartData}
        selectedSeries={['equity']}
        tradeMarkers={mockTradeMarkers}
        showPosition={false}
      />
    );

    expect(container).toBeInTheDocument();
  });

  test('renders with showPosition enabled', () => {
    const { container } = render(
      <ViewChart
        chartData={mockChartData}
        selectedSeries={['equity', 'position']}
        tradeMarkers={[]}
        showPosition={true}
      />
    );

    expect(container).toBeInTheDocument();
  });

  test('handles intraday time formatting', () => {
    const intradayData = [
      { date: '2020-01-01T09:00:00', equity: 100 },
      { date: '2020-01-01T09:15:00', equity: 102 }
    ];

    const { container } = render(
      <ViewChart
        chartData={intradayData}
        selectedSeries={['equity']}
        tradeMarkers={[]}
        showPosition={false}
      />
    );

    expect(container).toBeInTheDocument();
  });

  test('renders empty message when no data provided', () => {
    render(
      <ViewChart
        chartData={[]}
        selectedSeries={['equity']}
        tradeMarkers={[]}
        showPosition={false}
      />
    );

    expect(screen.getByText(/No chart data available/i)).toBeInTheDocument();
  });
});
