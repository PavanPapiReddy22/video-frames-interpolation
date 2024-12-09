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