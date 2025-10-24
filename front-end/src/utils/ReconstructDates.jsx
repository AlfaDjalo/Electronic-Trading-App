// utils/reconstructDates.js
export function reconstructTestDates(rawData, split = [0.8, 0.1, 0.1]) {
  if (!rawData || rawData.length === 0) return [];

  const total = rawData.length;
  const trainEnd = Math.floor(total * split[0]);
  const valEnd = Math.floor(total * (split[0] + split[1]));

  // Extract the test subset's actual dates
  const testDates = rawData.slice(valEnd).map(d => d.date);
  return testDates;
}
