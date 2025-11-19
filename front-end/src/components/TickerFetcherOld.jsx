import { useState, useEffect } from "react";

function TickerFetcher({ category }) {
  const [tickers, setTickers] = useState([]);

  useEffect(() => {
    async function fetchTickers() {
      if (!category) return;
      try {
        let url;
        if (category === "australian") {
          url = "https://en.wikipedia.org/wiki/S%26P/ASX_200";
        } else if (category === "us") {
          url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies";
        }

        const response = await fetch(url);
        const html = await response.text();

        // Use DOMParser (built into browsers) to parse the table
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, "text/html");

        let tickerList = [];
        if (category === "australian") {
          // ASX200 tickers are in the 3rd table on the page
          const table = doc.querySelectorAll("table")[2];
          tickerList = Array.from(table.querySelectorAll("tbody tr td:first-child"))
            .map(td => td.textContent.trim());
        } else if (category === "us") {
          // S&P500 tickers are in the first table
          const table = doc.querySelector("table");
          tickerList = Array.from(table.querySelectorAll("tbody tr td:first-child"))
            .map(td => td.textContent.trim());
        }

        setTickers(tickerList);
      } catch (err) {
        console.error("❌ Error fetching tickers:", err);
        setTickers([]);
      }
    }

    fetchTickers();
  }, [category]);

  return (
    <select>
      {tickers.map(t => (
        <option key={t} value={t}>
          {t}
        </option>
      ))}
    </select>
  );
}
