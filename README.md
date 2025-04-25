# About

The code implements a Continuous Variable Quantum Layer using Gaussian operations, integrating PyTorch with PennyLane for quantum computations. It processes quantum states using displacement, squeezing, and beam splitting.


## Getting Started

```bash
conda create -f requirements.yaml
conda activate qnn4pde
```

### Training Models


```bash
conda activate QCPINN

# Cavity
python -m src.poisson.cavity_hybrid_trainer

# Helmholtz
python -m src.poisson.helmholtz_hybrid_trainer

# Cavity-Hybrid
python -m src.poisson.klein_gordon_hybrid_trainer

# Wave
python -m src.poisson.wave_hybrid_trainer

# Diffusion
python -m src.poisson.diffusion_hybrid_trainer

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

If you find this work useful, please consider citing:

```bibtex
@article{farea2025qcpinn,
  title={QCPINN: Quantum Classical Physics-Informed Neural Networks for Solving PDEs},
  author={Farea, Afrah and Khan, Saiful and Celebi, Mustafa Serdar},
  journal={arXiv preprint arXiv:2503.16678},
  year={2025}
}
```

