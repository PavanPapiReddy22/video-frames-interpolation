import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{video_path}'.")
        return

    # Initialize frame count
    frame_count = 0

    # Read until video is completed
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if no frame is read
        if not ret:
            break

        # Save frame as an image
        frame_count += 1
        # Use the video name and frame count to generate unique filenames
        video_name = os.path.splitext(os.path.basename(video_path))[0]  # Get video name without extension
        frame_filename = os.path.join(output_folder, f"{video_name}_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

    # Release the video capture object
    cap.release()

    print(f"Frames extracted from '{video_path}': {frame_count}")

if __name__ == "__main__":
    # Input folder containing videos
    videos_folder = r"videos"

    # Output folder to save frames
    output_folder = "frames"

    # Iterate over each video file in the videos folder
    for video_file in os.listdir(videos_folder):
        if video_file.endswith(".mp4"):  # Assuming all video files are .mp4
            video_path = os.path.join(videos_folder, video_file)

            # Call the function to extract frames from this video
            extract_frames(video_path, output_folder)


def load_image(image_path, target_size):
    """
    Loads an image from a file, resizes it, and normalizes it.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0  # Normalize pixel values
    return image

def save_image(image, output_path):
    """
    Saves an image to the specified path, converting it back to 8-bit format.
    """
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image)

def interpolate_frames(input_folder, output_folder, model, target_size):
    """
    Processes frames in the input folder, interpolates using the model, and saves the results.
    
    Args:
        input_folder: Path to the folder containing input frames.
        output_folder: Path to the folder where output frames will be saved.
        model: Trained TensorFlow model for frame interpolation.
        target_size: Tuple (width, height) to resize frames.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all frame filenames, sorted numerically
    frame_files = sorted(os.listdir(input_folder), key=lambda x: int(os.path.splitext(x)[0]))
    
    for i in range(0, len(frame_files) - 1, 2):
        # Load the first and second frames
        frame1_path = os.path.join(input_folder, frame_files[i])
        frame2_path = os.path.join(input_folder, frame_files[i + 1])
        
        frame1 = load_image(frame1_path, target_size)
        frame2 = load_image(frame2_path, target_size)
        
        # Predict the middle frame using the model
        frame1_expanded = np.expand_dims(frame1, axis=0)
        frame2_expanded = np.expand_dims(frame2, axis=0)
        middle_frame = model.predict([frame1_expanded, frame2_expanded])[0]
        
        # Save the first and second frames to the output folder
        save_image(frame1, os.path.join(output_folder, frame_files[i]))
        save_image(frame2, os.path.join(output_folder, frame_files[i + 1]))
        
        # Calculate the name for the middle frame
        frame1_number = int(os.path.splitext(frame_files[i])[0])
        frame2_number = int(os.path.splitext(frame_files[i + 1])[0])
        middle_frame_number = (frame1_number + frame2_number) // 2
        middle_frame_name = f"{middle_frame_number:03d}.png"  # Zero-padded name
        
        # Save the middle frame
        save_image(middle_frame, os.path.join(output_folder, middle_frame_name))
        print(f"Processed frames {frame_files[i]} and {frame_files[i + 1]}, saved {middle_frame_name}")

# Load the model
model = load_model("model_colab1.h5", compile=False)
model.compile(optimizer="adam", loss="mse")

# Example usage
input_folder = r"frames"
output_folder = r"outputs"
target_size = (256, 256)  # Resize frames to match the model's input size

# Interpolate frames using the loaded model
interpolate_frames(input_folder, output_folder, model, target_size)


def images_to_video(image_folder, output_video_path, fps):
    """
    Combines images from a folder into a video.
    
    Args:
        image_folder (str): Path to the folder containing images.
        output_video_path (str): Path to save the output video file.
        fps (int): Frames per second for the output video.
    """
    # Get all image files in the folder, sorted by name
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(os.path.splitext(x)[0])  # Sort numerically based on filename
    )

    if not image_files:
        print("No images found in the specified folder.")
        return

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each image to the video
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read {image_file}. Skipping.")
            continue
        video_writer.write(image)

    # Release the video writer
    video_writer.release()
    print(f"Video saved to {output_video_path}")

# Example usage
image_folder = r"outputs"  # Path to the folder containing images
output_video_path = r"output_video.mp4"  # Path to save the output video
fps = 30  # Frames per second

images_to_video(image_folder, output_video_path, fps)

