import json

class FeatureSetManager:
    # FEATURE_SETS_FILE = "c:\\Users\\David\\Projects\\Electronic Trading App\\data\\feature_sets.json"

    def __init__(self, filepath):
        self.filepath = filepath
        self.feature_set_dictionary = {}
        self.load_feature_sets()

    def load_feature_sets(self):
        """
        Load feature sets from the JSON file.

        Returns:
            dict: A dictionary of feature sets.
        """
        print(f"Loading feature sets from {self.filepath}")
        try:
            with open(self.filepath, "r") as file:
                self.feature_set_dictionary = json.load(file)
                # return json.load(file)
        except FileNotFoundError:
            return {}

    def save_feature_sets(self):
        """
        Save the current feature sets to the JSON file, ensuring all required fields are present.
        """
        print(f"Saving feature sets to {self.filepath}")
        for name, feature_set in self.feature_set_dictionary.items():
            # Ensure required fields are present
            if "data_type" not in feature_set:
                feature_set["data_type"] = "daily"  # Default to "daily" if not specified
            if "features" not in feature_set:
                feature_set["features"] = []
            if "target" not in feature_set:
                feature_set["target"] = None  # Ensure "target" field exists
        with open(self.filepath, "w") as file:
            json.dump(self.feature_set_dictionary, file, indent=4)

    def get_feature_set(self, name):
        """
        Retrieve a specific feature set by name.

        Args:
            name (str): The name of the feature set.

        Returns:
            dict: The feature set dictionary, or None if not found.
        """
        return self.feature_set_dictionary.get(name)

    def add_feature_set(self, name, data):
        """
        Add a new feature set, ensuring required fields are included.

        Args:
            name (str): The name of the feature set.
            data (dict): The feature set data.
        """
        if "data_type" not in data:
            data["data_type"] = "daily"  # Default to "daily"
        if "features" not in data:
            data["features"] = []
        self.feature_set_dictionary[name] = data
        self.save_feature_sets()

    def update_feature_set(self, name, data, new_name=None):
        """
        Update an existing feature set or save it as a new feature set, ensuring required fields are included.

        Args:
            name (str): The name of the feature set to update.
            data (dict): The updated feature set data.
            new_name (str, optional): The new name for the feature set. If provided, the original feature set will remain unchanged.
        """
        if "data_type" not in data:
            data["data_type"] = self.feature_set_dictionary.get(name, {}).get("data_type", "daily")  # Retain existing or default to "daily"
        if "features" not in data:
            data["features"] = []
        if "target" not in data:
            data["target"] = self.feature_set_dictionary.get(name, {}).get("target", None)  # Retain existing target if not provided
        if new_name and new_name != name:
            if new_name in self.feature_set_dictionary:
                raise ValueError(f"Feature set '{new_name}' already exists.")
            self.feature_set_dictionary[new_name] = data
        elif name in self.feature_set_dictionary:
            self.feature_set_dictionary[name] = data
        else:
            raise ValueError(f"Feature set '{name}' does not exist.")
        self.save_feature_sets()

    def delete_feature_set(self, name):
        """
        Delete a feature set by name.

        Args:
            name (str): The name of the feature set.
        """
        if name in self.feature_set_dictionary:
            del self.feature_set_dictionary[name]
            self.save_feature_sets()  # Save changes to the JSON file
        else:
            raise ValueError(f"Feature set '{name}' does not exist.")

    def get_feature(self, feature_set_name, feature_name):
        """
        Retrieve a specific feature from a feature set.

        Args:
            feature_set_name (str): The name of the feature set.
            feature_name (str): The name of the feature.

        Returns:
            dict: The feature details, or None if not found.
        """
        feature_set = self.feature_set_dictionary.get(feature_set_name)
        if feature_set:
            for feature in feature_set.get("features", []):
                if feature.get("name") == feature_name:
                    return feature
        return None

    def set_feature(self, feature_set_name, feature_name, feature_data):
        """
        Update or add a feature in a feature set.

        Args:
            feature_set_name (str): The name of the feature set.
            feature_name (str): The name of the feature.
            feature_data (dict): The feature data to update or add.
        """
        feature_set = self.feature_set_dictionary.get(feature_set_name)
        if feature_set:
            features = feature_set.get("features", [])
            for feature in features:
                if feature.get("name") == feature_name:
                    feature.update(feature_data)
                    break
            else:
                features.append({"name": feature_name, **feature_data})
            feature_set["features"] = features
            self.save_feature_sets()
        else:
            raise ValueError(f"Feature set '{feature_set_name}' does not exist.")

    def get_feature_sets_by_data_type(self, data_type):
        """
        Retrieve all feature set names filtered by a specific data_type.

        Args:
            data_type (str): The data_type to filter by.

        Returns:
            list: A list of feature set names with the specified data_type.
        """
        return [name for name, fs in self.feature_set_dictionary.items() if fs.get("data_type") == data_type]
