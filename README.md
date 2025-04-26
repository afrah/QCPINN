# QCPINN: Quantum-Classical Physics-Informed Neural Networks

[![arXiv](https://img.shields.io/badge/arXiv-2503.16678-b31b1b.svg)](https://arxiv.org/abs/2503.16678)

Source code of QCPINN described in the paper: [QCPINN: Quantum-Classical Physics-Informed Neural Networks for Solving PDEs](https://arxiv.org/abs/2503.16678).


---


## Getting Started

### Prerequisites

[Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or any other Python environment.

### Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/afrah/QCPINN.git
cd QCPINN
conda env create -f qcpinn.yaml
conda activate qcpinn
```

### Training Models

Train models for different PDEs using the following commands:

```bash
# Cavity
python -m src.trainer.cavity_hybrid_trainer

# Helmholtz
python -m src.trainer.helmholtz_hybrid_trainer

# Klein-Gordon
python -m src.trainer.klein_gordon_hybrid_trainer

# Wave
python -m src.trainer.wave_hybrid_trainer

# Diffusion
python -m src.trainer.diffusion_hybrid_trainer
```

Jupyter notebooks for training, testing, and visualization are in `src/notebooks/`.

> **Note:** I used VS Code with the Jupyter extension for working on the notebooks.

## Inference

After training, generate plots and evaluate results:

```bash
# Cavity
python -m src.contour_plots.cavity_hybrid_plotting

# Helmholtz
python -m src.contour_plots.helmholtz_hybrid_plotting

# Klein-Gordon
python -m src.contour_plots.klein_gordon_hybrid_plotting

# Wave
python -m src.contour_plots.wave_hybrid_plotting

# Diffusion
python -m src.contour_plots.diffusion_hybrid_plotting
```

## Comparision  of the amplitude and angle encodings

```bash
# Cavity
python -m src.testing.cavity_test

# Helmholtz
python -m src.testing.helmholtz_test
```


Output plots and data will be saved in the appropriate results directory.

## Project Structure

```
QCPINN/
├── data/               # Cavity datasets from simulation
├── models/             # Saved models from training
├── qcpinn.yaml         # Conda environment file
└── src/
    ├── contour_plots/  # Plotting functions
    ├── data/           # Data generator
    ├── nn/             # Neural network modules
    ├── notebooks/      # Jupyter notebooks (training, testing, visualization)
    ├── trainer/        # Training scripts
    └── utils/          # Utility functions and helpers
```

> See the `src/notebooks/` folder for hands-on examples and further documentation.

## Support

If you encounter issues or have questions, please [open an issue](https://github.com/afrah/QCPINN/issues).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT [LICENSE](LICENSE)

## References

If you find this work useful, please consider citing:

```bibtex
@article{farea2025qcpinn,
  title={QCPINN: Quantum Classical Physics-Informed Neural Networks for Solving PDEs},
  author={Farea, Afrah and Khan, Saiful and Celebi, Mustafa Serdar},
  journal={arXiv preprint arXiv:2503.16678},
  year={2025}
}
```
