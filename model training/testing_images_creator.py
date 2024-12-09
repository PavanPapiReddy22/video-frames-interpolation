import os
import cv2
import random

def create_random_training_samples(frame_folder, output_folder, num_samples_per_video=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each folder containing video frames
    for video_folder in os.listdir(frame_folder):
        video_frames_folder = os.path.join(frame_folder, video_folder)
        frames = sorted(os.listdir(video_frames_folder))
        num_frames = len(frames)

        # Create random training samples for each video
        for _ in range(num_samples_per_video):
            # Randomly select a middle frame index
            middle_frame_index = random.randint(1, num_frames - 2)

            left_frame_path = os.path.join(video_frames_folder, frames[middle_frame_index - 1])
            right_frame_path = os.path.join(video_frames_folder, frames[middle_frame_index + 1])
            middle_frame_path = os.path.join(video_frames_folder, frames[middle_frame_index])

            training_sample_folder = os.path.join(output_folder, f"{video_folder}_training_sample_{middle_frame_index:04d}")
            os.makedirs(training_sample_folder, exist_ok=True)

            cv2.imwrite(os.path.join(training_sample_folder, "left_frame.jpg"), cv2.imread(left_frame_path))
            cv2.imwrite(os.path.join(training_sample_folder, "right_frame.jpg"), cv2.imread(right_frame_path))
            cv2.imwrite(os.path.join(training_sample_folder, "middle_frame.jpg"), cv2.imread(middle_frame_path))

if __name__ == "__main__":
    # Path to the folder containing video frame folders
    frame_folder = r"C:\Users\papir\Desktop\DSC 550\frames"

    # Output folder to save training samples
    training_output_folder = "training_samples"

    # Number of random samples to take per video
    num_samples_per_video = 2

    # Call the function to create random training samples
    create_random_training_samples(frame_folder, training_output_folder, num_samples_per_video)
