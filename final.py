import subprocess

# Paths to your scripts
scripts = [
    "D:/Vaibhav Hacker/Desktop/MTP/feature_extractor_rgb.py",
    "D:/Vaibhav Hacker/Desktop/MTP/feature_extractor_flow.py",
    "D:/Vaibhav Hacker/Desktop/MTP/feature_combiner.py"
]

for script in scripts:
    print(f" Running {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    
    print(f"Output of {script}:\n{result.stdout}")
    
    if result.stderr:
        print(f"Error in {script}:\n{result.stderr}")
    print("="*50)
