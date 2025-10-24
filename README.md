# DistributionalAutoencodersScore
Official code repository for Distributional Autoencoders Know the Score, NeurIPS 2025.  

Besides software from `pypi` installed via  `requirement.txt`, this repository depends on the following packages, which can be installed from their respective GitHub repositories:
- [Distributional Principal Autoencoder](https://github.com/xwshen51/DistributionalPrincipalAutoencoder)
- [mlcolvar](https://github.com/luigibonati/mlcolvar)

Their respective licenses are reproduced in the `third_party_licenses` folder.

# Structure
- `data` - datasets used in the experiments
- `exp` - the experiments scripts and notebooks
    * `Gaussian_score.ipynb` - reproduces Figure 1
    * `score_alignment.py` - reproduces Table 1
    * `MB.ipynb` - reproduces Figure 2
    * `MFEP_comparisons.py` - reproduces Table 2 and Figures 6, 7
    * `train_indep.py` - trains the basic models for Table 3
    * `train_swiss.py` - trains the Swiss-roll models for Table 3
    * `train_scurve.py` - trains the S-curve models for Table 3  
    * `train_scurve.sh `, `train_indep.sh`, `train_swiss.sh` - bash scripts to train the models for Table 3
    * `Indep-deterministic.ipynb` - reproduces Table 3
    * `run_CRT_linear.py` - performs the CRT experiment in Section 4.2  
    * `Indep-extra.ipynb` - reproduces Table 6
  
- `utils` - utility functions (load the module onto your path)
    * `mfep_utils.py` - utility functions for MFEP experiments
    * `plot_utils.py` - plotting utilities (some adapted from `mlcolvar`) 

## Citing
If you find this code useful in your research, please consider citing the following paper:

```
@inproceedings{leban2025distributionalautoencodersknowscore,
      title={Distributional Autoencoders Know the Score},
      author={Andrej Leban},
      year={2025},
      booktitle = {Advances in Neural Information Processing Systems},
}
```



