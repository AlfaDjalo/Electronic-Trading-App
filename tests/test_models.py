import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np

from models import ModelHandler

def test_model_handler_initialization(sample_stock_data, sample_params):
    """Test initialization of ModelHandler"""
    model_handler = ModelHandler(
        data=sample_stock_data,
        params=sample_params
    )
    
    data = model_handler.get_data()
    params = model_handler.get_params()
    assert data["close"] is not None
    assert params["TestModel"] is not None
    # assert params["num_units"] is not None
