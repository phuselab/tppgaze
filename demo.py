import torch
from tppgaze.tppgaze import TPPGaze
from tppgaze.utils import visualize_scanpaths

# Path to the configuration file
cfg_path = "data/config.yaml"

# Path to the trained model
checkpoint_path = "data/model_transformer.pth"

# Path to the image to generate scanpaths for
img_path = "data/1009.jpg"

# Device to run the model on
device = "cuda" if torch.cuda.is_available() else "cpu"

# Number of simulations to generate in parallel
n_simulations = 5

# Duration of each simulated scanpath in seconds
sample_duration = 2.0

# Initialize the model
model = TPPGaze(cfg_path, checkpoint_path, device)

# Load the trained model
model.load_model()

# Generate scanpaths for the image.
# The output is a list of scanpaths, where each scanpath is a 2D numpy array of shape (n_fixations, 3) with columns [x, y, fix_duration].
scanpaths = model.generate_predictions(img_path, sample_duration, n_simulations)

# Visualize the scanpaths
visualize_scanpaths(img_path, scanpaths)