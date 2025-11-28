<h1 align='center'>Distributional Autoencoders Know the Score</h1>
<div align='center'>
    <a href='https://andleb.netlify.app/' target='_blank'>Andrej Leban</a>
</div>
<div align='center'>
Department of Statistics, University of Michigan.
</div>
<div align='center'>
    <a href='https://openreview.net/pdf?id=5RIop1E1ga'><img src='https://img.shields.io/badge/Paper-NeurIPS2025-red'></a>
    <a href='https://arxiv.org/abs/2502.11583'><img src='https://img.shields.io/badge/Paper-ArXiv-Green'></a>
</div>  





Official code repository for [Distributional Autoencoders Know the Score](https://neurips.cc/virtual/2025/poster/119870), NeurIPS 2025.  

Besides software from `pypi` installed via  `requirements.txt`, this repository depends on the following packages:
- [Distributional Principal Autoencoder](https://github.com/xwshen51/DistributionalPrincipalAutoencoder)
- [mlcolvar](https://github.com/luigibonati/mlcolvar)
- [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE)

For code that was modified, the respective licenses are reproduced in the `third_party_licenses` folder.


## Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/andleb/DistributionalAutoencodersScore
   cd DistributionalAutoencodersScore
   ```
   
2. Install general dependencies:  
  ```bash
   pip install -r requirements.txt    
   ```

3. Install the Distributional Principal Autoencoder, mlcolvar, and PyTorch-VAE dependencies.

   **Option A:** Install from PyPI:
   ```bash
   pip install DistributionalPrincipalAutoencoder
   pip install mlcolvar
   # NOTE: PyTorch-VAE is not on PyPI, clone it into src/ as shown in Option B!
   ```

   **Option B:** Clone them locally, for example as submodules in the `src` folder:
   ```bash
   mkdir -p src
   cd src
   git submodule add git@github.com:xwshen51/DistributionalPrincipalAutoencoder.git
   git submodule add git@github.com:luigibonati/mlcolvar.git
   git submodule add git@github.com:AntixK/PyTorch-VAE.git
   cd ..
   ```

4. If needed, add the root of this repository to your `PYTHONPATH`:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```



## Structure and results

All paths below are relative to the repository root. Unless otherwise noted, we assume the repository root is on your `PYTHONPATH`. The results can be reproduced by running the files in the `exp/` folder.

For a quickstart, the `exp/Gaussian_score.ipynb` notebook is probably the best self-contained example to start with.

The structure of the repository is as follows:

- `data/` - datasets used in the experiments
- `exp/` - the experiment scripts and notebooks
- `data` - datasets used in the experiments
- `exp` - the experiments scripts and notebooks
    * `MB.ipynb` - reproduces Figure 2
    * `MFEP_comparisons.py` - reproduces Table 2 and Figures 6, 7
    * `train_indep.py` - trains the basic models for Table 3
    * `run_train_simple.sh` - bash script to train multiple models in parallel
    * `Indep-deterministic.ipynb` - analyzes conditional independence for deterministic extraneous latents (Table 3)
    * `train_swiss.py` - trains the Swiss-roll models for Table 3
    * `train_scurve.py` - trains the S-curve models for Table 3
    * `train_scurve.sh`, `run_train_simple.sh`, `train_swiss.sh` - bash scripts to train the models for Table 3
    * `Indep-deterministic.ipynb` - reproduces Table 3
    * `run_CRT_linear.py` - performs the CRT experiment in Section 4.2
    * `Indep-extra.ipynb` - reproduces Table 6
    * `mfep_utils.py` - utility functions for MFEP (Minimum Free Energy Path) experiments
- `utils` - utility functions (load the module onto your path)
    * `mfep_utils.py` - utility functions for MFEP experiments
    * `plot_utils.py` - plotting utilities (some adapted from `mlcolvar`)


```bibtex
@inproceedings{
  leban2025distributionalautoencodersknowscore,
  title={Distributional Autoencoders Know the Score},
  author={Andrej Leban},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://neurips.cc/virtual/2025/poster/119870}
}
```
