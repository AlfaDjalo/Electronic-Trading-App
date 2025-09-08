import { useEffect } from "react";

export default function FeatureSetLoader({ onLoaded }) {
  useEffect(() => {
    fetch("http://localhost:5000/api/feature_sets")
      .then(res => res.json())
      .then(data => {
        if (onLoaded) {
          onLoaded(data); // send feature sets up
        }
      })
      .catch(err => console.error("Error loading feature sets:", err));
  }, [onLoaded]);

  return null; // no UI, just loads feature sets
}