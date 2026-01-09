# GenAI Fine-Tuning Project

This repository demonstrates a modular workflow for **image generation fine-tuning and evaluation using Stable Diffusion and PyTorch**, focusing on clear, scalable structure for learning, experimentation, and demonstration.

It includes:
- Dataset preparation and loading
- Simulated fine-tuning pipeline structure
- Metrics logging (FID, IS, Precision, Recall)
- Modular and organized Python scripts

---

## Project Structure

genai_project/


├── config.py # Configuration settings

├── dataset_utils.py # Dataset loading and utilities

├── fine_tuning.py # Fine-tuning workflow

├── metrics.py # Metrics calculation and logging

├── utils.py # Utility functions for CLI and headers

├── main.py # Main pipeline orchestrator

└── README.md # Project documentation


---

## File Descriptions

- **config.py**  
  Contains:
  - Device configuration (CPU/GPU)
  - Random seeds for reproducibility
  - Hyperparameters for fine-tuning
  - Image generation settings
  - Dataset and output directory paths

- **dataset_utils.py**  
  Contains:
  - `CustomDataset` class for handling text-image pairs
  - Data transformations and normalization
  - Loader functions (`get_default_dataloader`, `get_demo_dataloader`) for easy integration

- **fine_tuning.py**  
  Contains:
  - Simulated fine-tuning workflow
  - Epoch and step-wise learning rate adjustments
  - Parameter updates and tracking
  - Checkpoint and status logging for transparency during runs

- **metrics.py**  
  Contains:
  - Simulated FID and IS score calculation structures
  - Precision and Recall calculation simulations
  - Logging of metrics into `logs/training_metrics.log` for persistent tracking

- **utils.py**  
  Contains:
  - Utility printing functions (`print_header`) for clear CLI section separation
  - Additional helpers for clean code separation

- **main.py**  
  The primary execution file orchestrating:
  - Dataset loading
  - Fine-tuning workflow invocation
  - Metrics logging
  - Multiple pipeline executions for extended demonstrations

---

## Features

✅ **Modular Design:** Each functionality is separated into its own file, aiding understanding and scalability.  
✅ **Simulated Fine-Tuning:** Useful for workflow understanding without heavy GPU usage while maintaining realistic structure for later integration.  
✅ **Metrics Logging:** Automatic logging into `logs/` for tracking performance over multiple runs.  
✅ **Ready for Real Data:** Replace placeholder datasets and simulation functions with real data pipelines and model training loops seamlessly.

---
## Output

<img width="577" height="422" alt="Image" src="https://github.com/user-attachments/assets/4d7d430e-6ab6-47e1-9732-4ff41bf03f73" />


<img width="557" height="410" alt="Image" src="https://github.com/user-attachments/assets/d224e81f-b5db-4310-800b-f1854b019428" />


<img width="517" height="408" alt="Image" src="https://github.com/user-attachments/assets/14c1109c-a9fd-4b31-915b-cdb88a2cab85" />


<img width="403" height="395" alt="Image" src="https://github.com/user-attachments/assets/5dabaedc-6039-473d-890a-a15f1e157a82" />


## How to Run

 **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/genai_project.git
   cd genai_project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch torchvision diffusers transformers tqdm numpy pillow
python main.py

