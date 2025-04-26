import json
import os

# Define input and output folders
input_folder = "labels"
output_folder = "ground_truth"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each JSON file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):  # Only process JSON files
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".json", ".txt"))

        # Load JSON file
        with open(input_path, "r") as file:
            data = json.load(file)

        # List to store lines
        lines = []

        # Read JSON and store task names based on frame ranges
        for subtask in data:
            task_name = subtask["subtask"]
            start_frame = subtask["start_frame"]
            end_frame = subtask["end_frame"]
            
            for _ in range(start_frame, end_frame + 1):
                lines.append(task_name)

        # Remove the first line to correct for 1-based indexing
        if lines:
            lines.pop(0)

        # Write the updated lines to the text file
        with open(output_path, "w") as file:
            file.write("\n".join(lines))

        print(f"Converted: {filename} → {output_path}")

print("All JSON files have been processed and saved in 'ground_truth'.")
