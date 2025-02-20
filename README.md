# TPP-Gaze: Modelling Gaze Dynamics in Space and Time with Neural Temporal Point Processes (WACV 2025)
[**Alessandro D'Amelio**](https://scholar.google.it/citations?user=chkawtoAAAAJ&hl=it),
[**Giuseppe Cartella**](https://scholar.google.com/citations?hl=en&user=0sJ4VCcAAAAJ),
[**Vittorio Cuculo**](https://scholar.google.it/citations?user=usEfqxoAAAAJ&hl=it&oi=ao),
[**Manuele Lucchi**](https://github.com/manuelelucchi),
[**Marcella Cornia**](https://scholar.google.com/citations?hl=en&user=DzgmSJEAAAAJ),
[**Rita Cucchiara**](https://scholar.google.com/citations?hl=en&user=OM3sZEoAAAAJ)
[**Giuseppe Boccignone**](https://scholar.google.it/citations?user=LqM0uJwAAAAJ&hl=it&oi=ao),


[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2410.23409)
[![Conference](https://img.shields.io/badge/WACV-2025-0076B9?labelColor=007582)]()

This is the **official repository** for the [**paper**](https://arxiv.org/abs/2410.23409) "*TPP-Gaze: Modelling Gaze Dynamics in Space and Time with Neural Temporal Point Processes*".

## Overview

<p align="center">
    <img src="figure.jpg">
</p>

>**Abstract**: <br>
> Attention guides our gaze to fixate the proper location of the scene and holds it in that location for the deserved amount of time given current processing demands, before shifting to the next one. As such, gaze deployment crucially is a temporal process. Existing computational models have made significant strides in predicting spatial aspects of observer's visual scanpaths (***where*** to look), while often putting on the background the temporal facet of attention dynamics (***when***). In this paper we present \tppgaze, a novel and principled approach to model scanpath dynamics based on Neural Temporal Point Process (TPP), that jointly learns the temporal dynamics of fixations position and duration, integrating deep learning methodologies with point process theory. We conduct extensive experiments across five publicly available datasets. Our results show the overall superior performance of the proposed model compared to state-of-the-art approaches.

## Installation and usage

1. Clone the repository

```bash
git clone https://github.com/phuselab/tppgaze
cd tppgaze
```

2. Install the required packages

```bash
pip install -r requirements.txt
```

3. To download the required `.pth` files, run the following script

```bash
bash download_models.sh
```

4. To run the demo, execute the following command

```bash
python demo.py
```

Content of `demo.py`

```python
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
```

## Citation
If you make use of our work, please cite our paper:

```bibtex
@inproceedings{damelio2025tppgaze,
  title={TPP-Gaze: Modelling Gaze Dynamics in Space and Time with Neural Temporal Point Processes},
  author={D'Amelio, Alessandro and Cartella, Giuseppe and Cuculo, Vittorio and Lucchi, Manuele and Cornia, Marcella and Cucchiara, Rita and Boccignone, Giuseppe},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2025}
}