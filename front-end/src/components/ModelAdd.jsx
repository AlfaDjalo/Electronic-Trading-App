import { useState } from "react";

import { ModelForm } from "./ModelForm";

export const ModelAdd = ({
  onAddModel,
  modelNames,
  featureSets
}) => {
  const [selectedModel, setSelectedModel] = useState(modelNames?.[0] || "");
  const [selectedFeatureSet, setSelectedFeatureSet] = useState(featureSets?.[0] || "");
  const [normalise, setNormalise] = useState(false);

  const handleAdd = (newModel) => {
    newModel.id = Date.now();
    onAddModel(newModel);
  };

  return (
    <div className="mb-6">
      <h3 className="text-xl font-bold mb-4">Add New Model</h3>
      <ModelForm
        initialValues={{}}
        onSubmit={handleAdd}
        modelNames={modelNames}
        featureSets={featureSets}
      />
    </div>
  );
};