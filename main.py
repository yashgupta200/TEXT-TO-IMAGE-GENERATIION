# main.py

from config import CFG
from dataset_utils import get_default_dataloader, get_demo_dataloader
from fine_tuning import full_fine_tuning_workflow
from metrics import simulate_metrics_logging
from utils import print_header

import time

def main_pipeline():
    print_header("Starting Main Pipeline")

    # Step 1: Load Data
    print("[INFO] Loading dataset using get_default_dataloader()...")
    dataloader = get_default_dataloader()
    time.sleep(0.5)

    print("[INFO] Loading additional dataset using get_demo_dataloader() for visualization...")
    demo_loader = get_demo_dataloader()
    time.sleep(0.5)

    # Step 2: Fine-Tuning Process
    print("[INFO] Starting fine-tuning workflow")
    full_fine_tuning_workflow()
    time.sleep(0.5)

    # Step 3: Metrics Logging
    print("[INFO] Calculating and logging evaluation metrics")
    simulate_metrics_logging()
    time.sleep(0.5)

    # Step 4: Completion Message
    print_header("Main Pipeline Execution Complete")
    print("[INFO] Model pipeline executed successfully")

def run_complete_pipeline_multiple_times():
    """
    Runs the main pipeline multiple times to simulate extended workflow processing.
    """
    print_header("Running Complete Pipeline Multiple Times")

    for i in range(2):
        print(f"[INFO] Iteration {i+1}/2 of Pipeline Execution")
        main_pipeline()
        time.sleep(1)

    print_header("All Pipeline Executions Completed")

if __name__ == "__main__":
    print_header("GENAI PROJECT PIPELINE STARTED")
    
    main_pipeline()

    # Optional repeated execution for extended demonstration
    run_complete_pipeline_multiple_times()

    print_header("GENAI PROJECT PIPELINE FINISHED")
