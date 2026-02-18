import os
from transformers import pipeline
from PIL import Image

# 1. SETUP
# Point to the folder where you saved the model in Step 2
MODEL_PATH = "./final_nsfw_classifier"

# Initialize the classification pipeline
# We use 'device=0' to run it on your GPU (much faster)
print("Loading model for inference...")
classifier = pipeline(
    "image-classification", 
    model=MODEL_PATH, 
    device=0  # Change to -1 if you don't have a GPU
)

def classify_nsfw(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found.")
        return

    # 2. RUN PREDICTION
    results = classifier(image_path)

    # 3. FORMAT RESULTS
    print(f"\n--- Results for: {os.path.basename(image_path)} ---")
    for result in results:
        # Convert score (0.938) to percentage (93.8%)
        percentage = result['score']
        label = result['label']
        print(f"{label:10}  {percentage:.2f}")

# --- TEST IT ---
# Change this to a path of an actual image on your computer
test_image = r"D:\IMG_20251122_004743.png"
classify_nsfw(test_image)