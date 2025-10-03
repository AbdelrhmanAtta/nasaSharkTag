import json

# How much to shift latitude by
LAT_SHIFT = 1   # change this value to what you need

# Load your JSON file
with open(r"ai\shark_datasets_no_probability.json", "r") as f:
    data = json.load(f)

# Loop through datasets and modify latitude
for dataset in data:
    for reading in dataset["readings"]:
        reading["location"]["latitude"] += LAT_SHIFT

# Save back the modified JSON
with open(r"ai\shark_datasets_no_probability_shifted.json", "w") as f:
    json.dump(data, f, indent=4)
