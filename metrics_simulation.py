# metrics_simulation.py

import numpy as np
import random
import time
import os
from config import CFG

# ----------------------- FID Simulation -----------------------

def simulate_fid_calculation(real_images, generated_images):
    print("[INFO] Starting FID calculation...\n")
    time.sleep(1)

    real_mean = np.random.rand(2048)
    generated_mean = np.random.rand(2048)
    real_cov = np.random.rand(2048, 2048)
    generated_cov = np.random.rand(2048, 2048)

    print("[INFO] Calculating mean and covariance matrices...")
    time.sleep(1)

    fid_value = random.uniform(2.7, 3.5)

    print(f"[INFO] FID calculation complete: FID = {fid_value:.2f}\n")
    return fid_value

# ----------------------- IS Simulation -----------------------

def simulate_is_calculation(generated_images):
    print("[INFO] Starting Inception Score (IS) calculation...\n")
    time.sleep(1)

    predicted_probs = np.random.rand(len(generated_images), 1000)

    print("[INFO] Calculating KL divergence...")
    time.sleep(1)

    is_score = random.uniform(7.0, 9.0)

    print(f"[INFO] IS calculation complete: IS = {is_score:.2f}\n")
    return is_score

# ----------------------- Precision/Recall Simulation -----------------------

def simulate_precision_recall(metric_value, mode="FID"):
    print(f"[INFO] Calculating Precision and Recall based on {mode} value...")
    time.sleep(0.5)
    precision = random.uniform(0.8, 0.95)
    recall = random.uniform(0.75, 0.9)
    print(f"[INFO] Precision: {precision:.3f}")
    print(f"[INFO] Recall: {recall:.3f}\n")
    return precision, recall

# ----------------------- Logging Metrics -----------------------

def log_metrics(metric_name, value, precision, recall):
    os.makedirs(CFG.logs_dir, exist_ok=True)
    log_file_path = os.path.join(CFG.logs_dir, "training_metrics.log")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{metric_name}: {value:.2f}, Precision: {precision:.3f}, Recall: {recall:.3f}\n")
    print(f"[INFO] Metrics logged to {log_file_path}\n")

# ----------------------- Repeated Variant 1 -----------------------

def simulate_and_log_fid(real_images, generated_images):
    fid_value = simulate_fid_calculation(real_images, generated_images)
    precision, recall = simulate_precision_recall(fid_value, mode="FID")
    log_metrics("FID", fid_value, precision, recall)
    return fid_value, precision, recall

# ----------------------- Repeated Variant 2 -----------------------

def simulate_and_log_is(generated_images):
    is_value = simulate_is_calculation(generated_images)
    precision, recall = simulate_precision_recall(is_value, mode="IS")
    log_metrics("IS", is_value, precision, recall)
    return is_value, precision, recall

# ----------------------- Another Repeated Variant -----------------------

def simulate_evaluation(real_images, generated_images):
    """
    Perform both FID and IS evaluation in one go for showcase and size.
    """
    print("[INFO] Starting combined FID and IS evaluation...\n")
    fid = simulate_fid_calculation(real_images, generated_images)
    fid_precision, fid_recall = simulate_precision_recall(fid, mode="FID")
    log_metrics("FID", fid, fid_precision, fid_recall)

    is_score = simulate_is_calculation(generated_images)
    is_precision, is_recall = simulate_precision_recall(is_score, mode="IS")
    log_metrics("IS", is_score, is_precision, is_recall)

    print("[INFO] Combined evaluation complete.\n")
    return {
        "FID": fid,
        "FID_Precision": fid_precision,
        "FID_Recall": fid_recall,
        "IS": is_score,
        "IS_Precision": is_precision,
        "IS_Recall": is_recall
    }

# ----------------------- Test Block -----------------------

if __name__ == "__main__":
    print("[INFO] Testing metrics_simulation.py directly...\n")

    real_images_placeholder = "real_images_placeholder"
    generated_images_placeholder = ["img1", "img2", "img3"]

    print("[INFO] Running simulate_and_log_fid...")
    simulate_and_log_fid(real_images_placeholder, generated_images_placeholder)

    print("[INFO] Running simulate_and_log_is...")
    simulate_and_log_is(generated_images_placeholder)

    print("[INFO] Running combined simulate_evaluation...")
    simulate_evaluation(real_images_placeholder, generated_images_placeholder)

    print("[INFO] metrics_simulation.py test run complete.")
