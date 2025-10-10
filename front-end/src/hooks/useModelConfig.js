import { useEffect, useState } from "react";

export const useModelConfig = () => {
  const [modelConfig, setModelConfig] = useState(null);

  useEffect(() => {
    fetch("http://localhost:5000/api/model_parameters")  // backend URL
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
        return res.json();
      })
      .then((data) => setModelConfig(data))
      .catch((err) => console.error("Error loading model config:", err));
  }, []);

  return modelConfig;
};


// import { useEffect, useState } from "react";

// export const useModelConfig = () => {
//   const [modelConfig, setModelConfig] = useState(null);

//   useEffect(() => {
//     fetch("/config/model_parameters.json")
//       .then((res) => res.json())
//       .then((data) => setModelConfig(data))
//       .catch((err) => console.error("Error loading model config:", err));
//   }, []);

//   return modelConfig;
// };