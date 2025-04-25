# About

The code implements a Continuous Variable Quantum Layer using Gaussian operations, integrating PyTorch with PennyLane for quantum computations. It processes quantum states using displacement, squeezing, and beam splitting.


## Getting Started

```bash
conda create -f requirements.yaml
conda activate qnn4pde
```

### Training Models


```bash
# Cavity
python src/poisson/cavity_train.py

# Helmholtz
python src/poisson/helmholtz_train.py

# Cavity-Hybrid
python src/poisson/cavity_hybrid_trainer.py

```

Note: Used VS Code Jupyter to run the Notebooks


### Inference


## Features

Code structure:

```

```

### Architecture
- Inherits from `nn.Module`
- Uses PennyLane's Gaussian device
- Implements separate circuits for X and P quadrature measurements
- Supports batch processing of quantum operations

### Parameters
1. Trainable parameters:
   - Displacements: shape (num_layers, num_qubits, 2)
   - Squeezing: shape (num_layers, num_qubits, 2)
   - Beamsplitter: shape (num_layers, num_qubits-1, 2)

2. Quantum Operations:
   - Displacement gates for state preparation
   - Squeezing gates with positive magnitude constraint
   - Beamsplitter gates with sigmoid-constrained transmittivity



## Technical Observations

1. **Quantum Circuit Design**
   - The circuit implements a layered architecture with alternating single-mode and two-mode operations
   - Each layer applies transformations in sequence: displacement → squeezing → beamsplitter
   - The measurement strategy uses both X and P quadratures for complete state information


## License

TODO 

## Support

Please open an issue for support.

## References

Please cite our paper as:

```bibtex
@article{farea2025pihcqnn,
  title={PI-HCQNN: Physics-Informed Hybrid Classical-Quantum Neural Networks for PDEs},
  author={Farea, Afrah and Khan, Saiful and others},
  journal={arXiv preprint arXiv:2504.10971},
  year={2025}
}
```

