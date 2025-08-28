// import { useState } from "react";

export const DataSummary = ({ dataInfo }) => {
    console.log(dataInfo.date_range);
    return (
        <div className="data-summary">
            <h3>Data Successfully Loaded</h3>
            <div className="summary-grid">
            <div className="summary-item">
                <span className="label">Features:</span>
                <span className="value">{dataInfo.num_features}</span>
            </div>
            <div className="summary-item">
                <span className="label">Observations:</span>
                <span className="value">{dataInfo.num_observations}</span>
            </div>
            <div className="summary-item">
                <span className="label">Date Range:</span>
                <span className="value">{dataInfo.date_range}</span>
            </div>
            <div className="summary-item">
                <span className="label">Features:</span>
                <span className="value">{dataInfo.feature_names.join(', ')}</span>
            </div>
            </div>
        </div>
    )
}
