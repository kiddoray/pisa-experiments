import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    hdri_path = "gs://kubric-public/assets/HDRI_haven/HDRI_haven.json"

    # download the JSON file from the cloud storage
    import subprocess
    subprocess.run(["gsutil", "cp", hdri_path, "./HDRI_haven.json"])

    hdri_data = load_json("./HDRI_haven.json")
    # showcase first 5 entries
    for entry in hdri_data[:5]:
        print(entry)
    # length of the dataset
    print(f"Total HDRI entries: {len(hdri_data)}")