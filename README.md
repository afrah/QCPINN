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

# Klein-Gordon
python -m src.poisson.klein_gordon_hybrid_trainer

# Wave
python -m src.poisson.wave_hybrid_trainer

# Diffusion
python -m src.poisson.diffusion_hybrid_trainer

```

Note: Used VS Code Jupyter to run the Notebooks


### Inference
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

## Features

Code structure:

```
.
├── README.md
├── data
│   └── cavity.mat
├── final_models
│   ├── 2025-02-06_19-25-14-069398
│   │   ├── 2025-02-06_19-25-14-069398
│   │   │   ├── circuit.pdf
│   │   │   ├── loss_history.pdf
│   │   │   ├── model.pth
│   │   │   ├── output.log
│   │   │   └── prediction.png
│   │   ├── circuit.pdf
│   │   ├── loss_history.pdf
│   │   ├── model.pth
│   │   ├── output.log
│   │   └── prediction.png
│   ├── 2025-02-06_19-28-34-814985
│   │   ├── circuit.pdf
│   │   ├── model.pth
│   │   └── output.log
│   ├── 2025-02-09_00-01-28-238904
│   │   ├── loss_history.pdf
│   │   ├── model.pth
│   │   ├── output.log
│   │   └── prediction.png
│   ├── 2025-02-21_11-27-26-796633
│   │   ├── circuit.pdf
│   │   ├── loss_history.pdf
│   │   ├── model.pth
│   │   └── prediction.png
│   ├── 2025-02-21_11-44-19-583365
│   │   ├── circuit.pdf
│   │   ├── loss_history.pdf
│   │   ├── model.pth
│   │   ├── output.log
│   │   └── prediction.png
│   ├── 2025-02-21_12-00-52-045180
│   │   ├── circuit.pdf
│   │   ├── loss_history.pdf
│   │   ├── model.pth
│   │   ├── output.log
│   │   └── tricontourf_10.pdf
│   ├── 2025-02-24_20-00-46-837506
│   │   ├── loss_history.pdf
│   │   ├── model.pth
│   │   ├── output.log
│   │   └── prediction.png
│   ├── 2025-02-25_17-01-13-323053
│   │   ├── loss_history.pdf
│   │   ├── model.pth
│   │   ├── output.log
│   │   └── prediction.png
│   ├── 2025-02-25_17-03-12-608017
│   │   ├── loss_history.pdf
│   │   ├── model.pth
│   │   └── output.log
│   └── 2025-02-25_17-21-36-221407
│       ├── model.pth
│       └── output.log
├── qcpinn-explicit.txt
├── results
│   └── circuit_case1.pdf
└── src
    ├── __pycache__
    │   └── utilities.cpython-38.pyc
    ├── contour_plots
    │   ├── cavity_hybrid_plotting.py
    │   ├── diffusion_hybrid_plotting.py
    │   ├── draw_losses.ipynb
    │   ├── helmholtz_hybrid_plotting.py
    │   ├── klein_gordon_hybrid_plotting.py
    │   ├── plotting_loss_history
    │   │   ├── cavity_test.py
    │   │   └── helmholtz_test.py
    │   └── wave_hybrid_plotting.py
    ├── data
    │   ├── cavity_dataset.py
    │   ├── diffusion_dataset.py
    │   ├── helmholtz_dataset.py
    │   ├── klein_gordon_dataset.py
    │   └── wave_dataset.py
    ├── nn
    │   ├── pde.py
    ├── notebooks
    │   ├── CV
    │   │   ├── cavity_hybrid_testing_CV_basic_cpu.ipynb
    │   │   └── helmholtz_hybrid_testing_CV_basic_cpu.ipynb
    │   ├── amplitude
    │   │   ├── cavity_hybrid_testing_DV_amplitude_alternate.ipynb
    │   │   ├── cavity_hybrid_testing_DV_amplitude_circuit19.ipynb
    │   │   ├── cavity_hybrid_testing_DV_amplitude_circuit5.ipynb
    │   │   ├── helmholtz_hybrid_testing_DV_amplitude_alternate.ipynb
    │   │   ├── helmholtz_hybrid_testing_DV_amplitude_circuit5.ipynb
    │   │   └── helmholtz_hybrid_testing_DV_amplitude_circuit9.ipynb
    │   ├── check_for_cuda.ipynb
    │   ├── classical
    │   │   ├── cavity_hybrid_testing_classical.ipynb
    │   │   └── helmholtz_hybrid_testing_classical.ipynb
    │   ├── gpu
    │   │   ├── cavity_hybrid_testing_CV_basic.ipynb
    │   │   ├── cavity_hybrid_testing_CV_enhanced.ipynb
    │   │   ├── helmholtz_hybrid_testing_CV_basic.ipynb
    │   │   └── helmholtz_hybrid_testing_CV_enahnced.ipynb
    │   ├── plotting
    │   │   ├── cavity_loss_history.ipynb
    │   │   ├── cavity_plot_loss_history.ipynb
    │   │   ├── helmholtz_loss_history.ipynb
    │   │   └── helmholtz_plot_loss_history.ipynb
    │   ├── testing
    │   │   ├── cavity_testing_angle_cir19.ipynb
    │   │   ├── diffusion_gordon_testing_angle_cir19.ipynb
    │   │   ├── helmholtz_testing_angle_cir19.ipynb
    │   │   ├── klein_gordon_testing_angle_cir19.ipynb
    │   │   ├── loss_history
    │   │   │   ├── cavity_test.ipynb
    │   │   │   └── helmholtz_test.ipynb
    │   │   └── wave_test_angle_cir19.ipynb
    │   └── training
    │       ├── Klein_gordon_hybrid_training copy.ipynb
    │       ├── Klein_gordon_hybrid_training_classic.ipynb
    │       ├── cavity_hybrid_testing_CV.ipynb
    │       ├── cavity_hybrid_training.ipynb
    │       ├── cavity_hybrid_training_amplitude.ipynb
    │       ├── diffusion_hybrid_training_classic.ipynb
    │       ├── diffusion_hybrid_training_quantum.ipynb
    │       ├── helmholtz_hybrid_testing_CV.ipynb
    │       ├── helmholtz_hybrid_training.ipynb
    │       ├── helmholtz_hybrid_training_amplitude.ipynb
    │       ├── wave_hybrid_training.ipynb
    │       └── wave_hybrid_training_classic.ipynb
    ├── poisson
    │   ├── CVNeuralNetwork.py
    │   ├── CVNeuralNetwork2.py
    │   ├── ECVQNN.py
    │   ├── cavity_train.py
    │   ├── classical_solver.py
    │   ├── cv_solver.py
    │   ├── cvquantum_layer.py
    │   ├── diffusion_train.py
    │   ├── dv_quantum_layer.py
    │   ├── dv_solver.py
    │   ├── helmholtz_train.py
    │   ├── klein_gordon_train.py
    │   └── wave_train.py
    └── utils
        ├── cavity_plot_contour.py
        ├── cavity_plot_prediction.py
        ├── cmap.py
        ├── color.py
        ├── common.py
        ├── error_metrics.py
        ├── functions.py
        ├── get_default_args.py
        ├── logger.py
        ├── plot_loss.py
        ├── plot_model_results.py
        ├── plot_prediction.py
        ├── plotting.py
        └── utilities.py
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

