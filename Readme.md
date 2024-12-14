# Project Instructions  

## Setting Up the Environment  

To ensure the project runs smoothly, follow these steps to set up a new Conda environment and install the required dependencies:  

### Step 1: Create a New Conda Environment  

1. Open **Anaconda Prompt**.  
2. Run the following command to create a new environment with Python 3.8:  

   ```bash  
   conda create --name <env_name> python=3.8  
   ```  

   Replace `<env_name>` with your desired environment name.  

3. Activate the newly created environment:  

   ```bash  
   conda activate <env_name>  
   ```  

### Step 2: Install Dependencies  

1. Ensure the `requirements.txt` file is in your project directory.  
2. Install all dependencies by running:  

   ```bash  
   pip install -r requirements.txt  
   ```  

### Step 3: Open Jupyter Notebook  

1. Navigate to the project directory:  

   ```bash  
   cd <project_folder>  
   ```  

   Replace `<project_folder>` with the path to your project folder.  

2. Launch Jupyter Notebook:  

   ```bash  
   jupyter notebook  
   ```  

3. Open the file `test.ipynb` in Jupyter Notebook.  

---

## Running the Project  

### Step 1: Prepare the Video Files  

- Copy the video file into the `videos` folder within your project directory. (one video file at a time)

### Step 2: Execute the Notebook  

1. In Jupyter Notebook, open `test.ipynb`.  
2. Run all the cells sequentially.  

### Step 3: Output  

- After successful execution, the processed video file will be saved in the same folder as the Jupyter Notebook (`test.ipynb`).  

---

## Additional Notes  

- Ensure you are using the newly created Conda environment when running Jupyter Notebook.  
- If you encounter any issues, verify that all dependencies are installed properly and that the video files are in the correct folder.  

---  

# Overview

# Video Frame Interpolation Using Transformer-Based Architecture

## Dataset

For training our model, we created a custom dataset by extracting frames from low-FPS videos and dividing them into individual images. Approximately 20 videos were processed, broken down into frames, and organized into separate folders.

### Training Sample Design

- **Input Frames:** Two consecutive frames (`Frame_t-1` and `Frame_t+1`).
- **Target Frame:** The intermediate frame (`Frame_t`) predicted by the model.

### Data Preparation

Each training sample consists of:

1. A pair of input frames: the preceding frame (`Frame_t-1`) and the succeeding frame (`Frame_t+1`).
2. The ground truth intermediate frame (`Frame_t`).

---

## Model Architecture

Our model predicts an intermediate frame between two consecutive input frames, using a convolutional neural network (CNN) inspired architecture.

### Key Components

#### 1. Input Layer

- **Input Frames:**
  - `Frame_t-1`: Captures spatial and motion context before the target frame.
  - `Frame_t+1`: Provides motion continuity and future scene progression.

#### 2. Convolutional Layers

- **Feature Extraction:** Hierarchical layers capture low-level (edges, textures), mid-level (patterns, shapes), and high-level (global structures) features.
- **Concatenation:** Features from the two input frames are merged to form a unified representation.

#### 3. Output Layer

- **Predicted Frame:** Outputs a full-resolution RGB frame (`Frame_t`).
- **Activation Function:** Sigmoid activation ensures pixel values are normalized between 0 and 1.

### Layer Summary

| Layer Name        | Input Shape     | Output Shape    | Parameters |
|-------------------|-----------------|-----------------|------------|
| Left Input        | (256, 256, 3)  | (256, 256, 3)  | 0          |
| Right Input       | (256, 256, 3)  | (256, 256, 3)  | 0          |
| Conv2D (1)        | (256, 256, 3)  | (256, 256, 32) | 896        |
| Conv2D (2)        | (256, 256, 32) | (256, 256, 64) | 18,496     |
| Conv2D (3)        | (256, 256, 64) | (256, 256, 128)| 73,856     |
| Concatenation     | (256, 256, 128)| (256, 256, 256)| 0          |
| Conv2D (4)        | (256, 256, 256)| (256, 256, 256)| 590,080    |
| Conv2D (5)        | (256, 256, 256)| (256, 256, 128)| 295,040    |
| Conv2D (6)        | (256, 256, 128)| (256, 256, 64) | 73,792     |
| Conv2D (7)        | (256, 256, 64) | (256, 256, 32) | 18,464     |
| Output Layer      | (256, 256, 32) | (256, 256, 3)  | 867        |
| **Total Parameters** |                 |                 | **1,071,491** |

---

## Model Training

### 1. Loss Function

We used **Mean Squared Error (MSE)** to measure the pixel-wise error between predicted and ground truth frames

### 2. Optimization

The **Adam Optimizer** was used for efficient gradient updates. It combines momentum and adaptive learning rates for faster convergence.

### 3. Training Strategy

- **Epochs:** 75.
- **Batch Size:** 16.
- **Validation:** A portion of the dataset was used for validation to monitor performance and prevent overfitting.
- **Learning Rate Scheduling:** Reduced learning rate upon plateau to refine predictions.

---

## Evaluation Metrics

### 1. Peak Signal-to-Noise Ratio (PSNR)

Measures the fidelity of the predicted frame:

- **High PSNR (>30 dB):** Excellent quality.
- **Moderate PSNR (20-30 dB):** Reasonable quality.
- **Low PSNR (<20 dB):** Poor quality.

### 2. Structural Similarity Index (SSIM)

Evaluates structural alignment between predicted and ground truth frames:

- Ranges from 0 (no similarity) to 1 (perfect match).

### Results

- **PSNR:** 25.06 dB.
- **SSIM:** 0.748.

---

## Key Features

- **Symmetric Architecture:** Ensures balanced feature extraction from both input frames.
- **Feature Fusion:** Combines spatial and temporal information via concatenation.
- **Temporal Consistency:** Smooth transitions between frames with minimal artifacts.
- **Efficiency:** Lightweight architecture with just over 1M parameters, suitable for real-time applications.

---

## Future Improvements

- **Perceptual Loss:** Incorporate perceptual loss functions to enhance structural quality.
- **Data Augmentation:** Improve generalization by including diverse scenarios.
- **Advanced Techniques:** Explore attention mechanisms or GAN-based approaches for higher-quality results.

---

## Applications

- **Video Upscaling:** Enhance video resolution.
- **Slow-Motion Generation:** Synthesize intermediate frames for smoother playback.
- **Frame Rate Enhancement:** Improve streaming or gaming experiences.

---

## Conclusion

Our Transformer-based model for video frame interpolation achieves a balance between quality and computational efficiency. With further enhancements, it has potential applications in various real-world scenarios.


# Results
- The images on the left represent the ground truth, while the images on the right depict the predicted outputs.
- The ground truth images are preprocessed for comparison.

![image](https://github.com/user-attachments/assets/6309e623-306d-4ec4-af9a-2fdec816857d)


![image](https://github.com/user-attachments/assets/30d56420-21fa-4f0c-81a8-c5f78fd664a5)




  


