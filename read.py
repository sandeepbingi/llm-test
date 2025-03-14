import pickle
import json
import pandas as pd
import os
import pprint

# Function to load and print .pkl file in a readable format
def read_pickle_file(file_path):
    """Reads a .pkl file and prints its content in a readable format."""
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        return

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    print("\n🔹 Pickle File Content:")
    pprint.pprint(data)  # Pretty print the data

# Function to convert a .pkl file to JSON format
def pickle_to_json(file_path, output_json=None):
    """Converts a .pkl file to a JSON file (human-readable)."""
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        return

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Convert set to list (JSON doesn't support sets)
    if isinstance(data, set):
        data = list(data)

    # If no output file provided, save as same name with .json extension
    if output_json is None:
        output_json = file_path.replace(".pkl", ".json")

    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"✅ Converted '{file_path}' to '{output_json}'")

# Function to read a pickle file containing a Pandas DataFrame
def read_pickle_dataframe(file_path):
    """Loads a .pkl file as a Pandas DataFrame and prints it."""
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        return

    try:
        df = pd.read_pickle(file_path)
        print("\n🔹 Pickle File Loaded as Pandas DataFrame:")
        print(df.head())  # Show the first 5 rows
    except Exception as e:
        print(f"❌ Error: Could not read DataFrame from '{file_path}': {e}")

# Function to batch convert all .pkl files in a directory to JSON
def batch_convert_pickle_to_json(directory):
    """Converts all .pkl files in the given directory to JSON."""
    if not os.path.exists(directory):
        print(f"❌ Error: Directory '{directory}' not found.")
        return

    pkl_files = [f for f in os.listdir(directory) if f.endswith(".pkl")]

    if not pkl_files:
        print("❌ No .pkl files found in the directory.")
        return

    print(f"\n🔹 Found {len(pkl_files)} .pkl files. Converting to JSON...")
    for file in pkl_files:
        pickle_to_json(os.path.join(directory, file))
    print("✅ Batch conversion complete!")

# Run script with example usage
if __name__ == "__main__":
    print("\n🔹 Pickle File Utility 🔹\n")
    
    # Example file paths (Change these as needed)
    pkl_file = "processed_files_QWE_logins.pkl"
    directory = "."  # Current directory

    # Uncomment the function you want to use:
    
    # View .pkl content
    # read_pickle_file(pkl_file)

    # Convert single .pkl to JSON
    # pickle_to_json(pkl_file)

    # Read a .pkl as a Pandas DataFrame
    # read_pickle_dataframe(pkl_file)

    # Convert all .pkl files in a directory to JSON
    # batch_convert_pickle_to_json(directory)
