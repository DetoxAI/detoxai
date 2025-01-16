import os

# This is for ensuring that the activations directory is created
__detoxai_root = os.getenv("DETOXAI_ROOT_PATH")
ACTIVATIONS_DIR = os.path.join(__detoxai_root, "activations")
os.makedirs(ACTIVATIONS_DIR, exist_ok=True)
