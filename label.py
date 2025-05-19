import cv2
import os
import json
from natsort import natsorted  # Import for natural sorting


def video_to_frames(video_path, frames_folder):
    """
    Extract frames from a video and save them as images in the specified folder.
    """
    os.makedirs(frames_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path} to {frames_folder}.")
    return frame_count


def load_frames(folder_path):
    return sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".jpg")])


def save_annotations(output_path, annotations):
    if annotations:  # Save only if there are annotations
        with open(output_path, 'w') as f:
            json.dump(annotations, f, indent=4)
        print(f"Annotations saved to {output_path}")
    else:
        print(f"No annotations to save for {output_path}")


def annotate(frame_paths, output_path):
    global stop_script  # Use a global variable to end the process immediately
    current_frame_idx = 0
    start_frame = 0
    annotations = []
    
    label_input = ""
    editing_label = False

    while not stop_script:  # Check if the script should stop
        frame = cv2.imread(frame_paths[current_frame_idx])
        display_text = f"Frame {current_frame_idx + 1}/{len(frame_paths)}"

        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Controls: . (Next), , (Previous), Enter (Label), ` (Save), \\ (Stop), ESC (Exit)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Current Label: {label_input}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if editing_label:
            cv2.putText(frame, "Enter Label (Press Enter when done):", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Frame Annotation Tool", frame)
        key = cv2.waitKey(0)

        if key == 27:  # ESC key
            print("Exiting annotation.")
            break
        elif key == ord("\\"):  # Stop annotating
            print("User requested to stop annotating.")
            stop_script = True
            break
        elif key == ord("."):  # Next frame
            current_frame_idx = min(current_frame_idx + 1, len(frame_paths) - 1)
        elif key == ord(","):  # Previous frame
            current_frame_idx = max(current_frame_idx - 1, 0)
        elif key == 13:  # Enter
            if not editing_label:
                editing_label = True
            else:
                if label_input.strip():
                    end_frame = current_frame_idx
                    annotations.append({
                        "subtask": label_input,
                        "start_frame": start_frame,
                        "end_frame": end_frame
                    })
                    print(f"Sub-task '{label_input}' recorded: start={start_frame}, end={end_frame}")
                    start_frame = end_frame + 1
                    label_input = ""
                    editing_label = False
                    current_frame_idx = min(current_frame_idx + 1, len(frame_paths) - 1)
        elif key == ord("`"):  # Save
            save_annotations(output_path, annotations)
            print("Progress saved.")
        elif editing_label:  # Label input
            if key == 8:  # Backspace
                label_input = label_input[:-1]
            elif key != 13:
                label_input += chr(key)

    if not stop_script:  # Save annotations only if the process wasn't stopped
        save_annotations(output_path, annotations)
    else:
        print("Annotation stopped. Not saving annotations for this video.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stop_script = False  # Initialize the global variable

    # Specify the folder containing videos
    vidType = 'mopping'
    videos_folder = "mopping/videos"
    frames_base_folder = "frames"
    annotations_base_folder = "labels"

    # Process each video in the videos folder
    for video_filename in natsorted(os.listdir(videos_folder)):  # Use natsorted for natural sorting
        if stop_script:  # Check if the user requested to stop
            print("Labelling stopped by the user.")
            break

        if not video_filename.endswith(('.mp4', '.avi')):
            continue

        video_path = os.path.join(videos_folder, video_filename)
        video_name = os.path.splitext(video_filename)[0]

        # Create the annotations file path
        annotations_file = os.path.join(annotations_base_folder, vidType, f"{video_name}.json")
        
        # Check if the JSON file already exists
        if os.path.exists(annotations_file):
            print(f"Skipping video {video_filename}, JSON file already exists.")
            continue  # Skip this video if annotations already exist

        # Create folders for frames
        frames_folder = os.path.join(frames_base_folder, vidType, video_name)
        os.makedirs(os.path.dirname(annotations_file), exist_ok=True)

        # Extract frames from the video
        print(f"Processing video: {video_filename}")
        video_to_frames(video_path, frames_folder)

        # Load frames for annotation
        frame_paths = load_frames(frames_folder)
        if not frame_paths:
            print(f"No frames found for {video_filename}.")
        else:
            annotate(frame_paths, annotations_file)

    print("Script finished.")