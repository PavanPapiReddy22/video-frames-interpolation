import cv2
import os

def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Initialize frame count
    frame_count = 0

    # Read until video is completed
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if no frame read
        if not ret:
            break

        # Save frame as an image
        frame_count += 1
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

    # Release the video capture object
    cap.release()

    print(f"Frames extracted: {frame_count}")

if __name__ == "__main__":
    # Path to the video file
    video_path = r"C:\Users\papir\Desktop\DSC 550\sample_video.mp4"

    # Output folder to save frames
    output_folder = "frames"

    # Call the function to extract frames
    extract_frames(video_path, output_folder)
