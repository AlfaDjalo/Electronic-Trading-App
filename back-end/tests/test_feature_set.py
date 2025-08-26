import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import json
from feature_set import FeatureSetManager

FEATURE_SETS_FILE = "c:\\Users\\David\\Projects\\Electronic Trading App\\data\\feature_sets.json"

@pytest.fixture
def feature_set_manager():
    """Fixture to create a FeatureSet instance with a temporary file."""
    temp_file = "temp_feature_sets.json"
    with open(temp_file, "w") as f:
        json.dump({}, f)
    manager = FeatureSetManager(FEATURE_SETS_FILE)
    manager.FEATURE_SETS_FILE = temp_file
    yield manager
    os.remove(temp_file)

def test_add_and_get_feature_set(feature_set_manager):
    """Test adding and retrieving a feature set."""
    feature_set_manager.add_feature_set("Test Set", {"data_type": "daily", "features": []})
    feature_set = feature_set_manager.get_feature_set("Test Set")
    assert feature_set is not None
    assert feature_set["data_type"] == "daily"
    assert feature_set["features"] == []

def test_update_feature_set(feature_set_manager):
    """Test updating an existing feature set."""
    feature_set_manager.add_feature_set("Test Set", {"data_type": "daily", "features": []})
    feature_set_manager.update_feature_set("Test Set", {"data_type": "intraday", "features": [{"name": "Feature 1"}]})
    feature_set = feature_set_manager.get_feature_set("Test Set")
    assert feature_set["data_type"] == "intraday"
    assert feature_set["features"][0]["name"] == "Feature 1"

def test_delete_feature_set(feature_set_manager):
    """Test deleting a feature set."""
    feature_set_manager.add_feature_set("Test Set", {"data_type": "daily", "features": []})
    feature_set_manager.delete_feature_set("Test Set")
    assert feature_set_manager.get_feature_set("Test Set") is None

def test_get_and_set_feature(feature_set_manager):
    """Test getting and setting a feature in a feature set."""
    feature_set_manager.add_feature_set("Test Set", {"data_type": "daily", "features": []})
    feature_set_manager.set_feature("Test Set", "Feature 1", {"input_data_fields": ["Close"], "function": "raw_data"})
    feature = feature_set_manager.get_feature("Test Set", "Feature 1")
    assert feature is not None
    assert feature["input_data_fields"] == ["Close"]
    assert feature["function"] == "raw_data"

def test_load_feature_sets(feature_set_manager):
    """Test loading feature sets."""
    feature_sets = feature_set_manager.load_feature_sets()
    assert isinstance(feature_sets, dict)

def test_save_feature_sets(feature_set_manager):
    """Test saving feature sets."""
    feature_set_manager.add_feature_set("Test Set", {"data_type": "daily", "features": []})
    feature_set_manager.save_feature_sets()
    with open(feature_set_manager.FEATURE_SETS_FILE, "r") as f:
        saved_data = json.load(f)
    assert "Test Set" in saved_data
    assert saved_data["Test Set"]["data_type"] == "daily"
    assert saved_data["Test Set"]["features"] == []
