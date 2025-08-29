import { useEffect, useState } from "react";

export const useModelConfig = () => {
  const [modelConfig, setModelConfig] = useState(null);

  useEffect(() => {
    fetch("/config/model_parameters.json")
      .then((res) => res.json())
      .then((data) => setModelConfig(data))
      .catch((err) => console.error("Error loading model config:", err));
  }, []);

  return modelConfig;
};