// import { useState } from "react";

export const DataSummary = ({ data }) => {
    console.log(data);
    return (
        <div className="data-summary">
            <h3>Data Successfully Loaded</h3>
            <div className="summary-grid">
            <div className="summary-item">
                <span className="label">Features:</span>
                <span className="value">{dataInfo.seriesNames.length}</span>
            </div>
            <div className="summary-item">
                <span className="label">Observations:</span>
                <span className="value">{dataInfo.timeSeriesData.length}</span>
            </div>
            <div className="summary-item">
                <span className="label">Features:</span>
                <span className="value">{dataInfo.seriesNames.join(', ')}</span>
            </div>
            </div>
        </div>
    )
}
