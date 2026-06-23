import pandas as pd
import numpy as np
import os

# ================= Configuration =================
input_path = "./weight.csv"
output_path = "./weight_soft.csv"

# Keep a large Sigma for smoothness, we will manually truncate later
SIGMA = 10.0
# Boost factor for the target class to ensure correct classification
BOOST = 1.5


# ================= Core Algorithm =================
def calculate_soft_label_top2(weight, phi_1, phi_2, sigma=10.0, boost=1.5):
    """
    Calculate soft labels dynamically based on dataset percentiles (phi_1, phi_2).
    """
    # --- 1. Determine hard label based on dynamic percentiles ---
    if weight < phi_1:
        cls_idx = 0
    elif phi_1 <= weight <= phi_2:
        cls_idx = 1
    else:
        cls_idx = 2

    # --- 2. Calculate absolute distance to boundaries ---
    dist_light = 0 if weight < phi_1 else weight - phi_1

    if phi_1 <= weight <= phi_2:
        dist_mid = 0
    elif weight < phi_1:
        dist_mid = phi_1 - weight
    else:
        dist_mid = weight - phi_2

    dist_heavy = 0 if weight > phi_2 else phi_2 - weight

    # --- 3. Gaussian scoring ---
    scores = np.array([
        np.exp(-(dist_light ** 2) / (2 * sigma ** 2)),
        np.exp(-(dist_mid ** 2) / (2 * sigma ** 2)),
        np.exp(-(dist_heavy ** 2) / (2 * sigma ** 2))
    ])

    # --- 4. Apply boost to target class ---
    scores[cls_idx] *= boost

    # ================= Keep Top-2 Only =================
    # Logic: Find the index with the minimum score and set it to 0.0
    # This ensures only two adjacent or most relevant intervals are kept
    min_idx = np.argmin(scores)
    scores[min_idx] = 0.0

    # --- 5. Re-normalize ---
    total = np.sum(scores)
    # Prevent division by zero
    if total == 0:
        final_probs = [0.0, 0.0, 0.0]
        final_probs[cls_idx] = 1.0
    else:
        final_probs = scores / total

    return pd.Series([final_probs[0], final_probs[1], final_probs[2], int(cls_idx)])


# ================= Execution =================
if __name__ == "__main__":
    if os.path.exists(input_path):
        print(f"Reading: {input_path}")
        df = pd.read_csv(input_path)

        # --- Dynamic Boundary Calculation ---
        # Calculate 33.3rd and 66.7th percentiles based on the empirical distribution
        phi_1 = df['weight'].quantile(0.333)
        phi_2 = df['weight'].quantile(0.667)

        print(f"Calculated Dynamic Boundaries: phi_1 = {phi_1:.2f} kg, phi_2 = {phi_2:.2f} kg")
        print("Generating Top-2 soft labels...")

        # Apply the function with dynamic percentiles
        df[['prob_light', 'prob_mid', 'prob_heavy', 'cls']] = df['weight'].apply(
            lambda x: calculate_soft_label_top2(x, phi_1, phi_2, sigma=SIGMA, boost=BOOST)
        )

        # --- Preview ---
        print("\n=== Preview ===")
        # Select specific samples around the dynamic boundaries for preview
        mask = ((df['weight'] >= phi_1 - 3) & (df['weight'] <= phi_1 + 3)) | \
               ((df['weight'] >= phi_2 - 3) & (df['weight'] <= phi_2 + 3))
        preview = df[mask].sort_values('weight').drop_duplicates('weight').head(8)
        print(preview[['ID', 'weight', 'prob_light', 'prob_mid', 'prob_heavy', 'cls']])

        df.to_csv(output_path, index=False)
        print(f"\nFile saved to: {output_path}")
    else:
        print("File not found")