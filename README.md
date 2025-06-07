# Solar Eclipse Model

*A Python package for simulating and fitting solar eclipse light curves.*

This Python package simulates and fits solar eclipse light curves using ephemerides and a transit modeling approach. It supports multi-model comparisons via MCMC sampling and produces plots for scientific analysis.


---

## 🚀 Features

- Simulate solar eclipses with `batman`-based light curve models
- Integrate Sun and Moon ephemerides from `astropy`
- Support for limb darkening and atmospheric extinction
- MCMC fitting using `emcee`
- YAML-configurable multi-model pipeline


---

## 🧪 Requirements

```bash
pip install numpy pandas matplotlib astropy batman-package emcee pyyaml corner sphinx sphinx_rtd_theme
```

---

## 📁 File Overview

| File                    | Purpose                                                            |
|-------------------------|--------------------------------------------------------------------|
| `config/config.yaml`    | YAML configuration for model parameters and execution              |
| `custom_models/`        | Contains core model implementations (e.g., `solareclipsemodel.py`) |
| `data/`                 | Directory for observational data files                             |
| `docs/`                 | Sphinx documentation source files                                  |
| `LICENSE`               | GNU LICENSE file                                                   | 
| `model_eclipse_mcmc.py` | Primary CLI script for running MCMC model fitting                  |
| `README.md`             | Project overview and quick start guide                             |
| `requirements.txt`      | Python package dependencies                                        |
| `utils/`                | Provides helper functions for MCMC analysis and plotting           |

---

## 🛠️ Running a Model

```bash
python model_eclips_mcmc.py --config config.yaml --model all
```

Run a single model:

```bash
python model_eclips_mcmc.py --config config.yaml --model linear
```

---

## 📊 Output

- CSV summary of MCMC fit
- Corner plots of posterior distributions
- Comparison plots of data vs. model

---

## 📚 Documentation

Build local docs using:

```bash
cd docs
make html
```

View in browser:

```bash
open _build/html/index.html
```

---

## 📦 Third-Party Libraries Used

This project makes use of the following open-source libraries:

- [`batman-package`](https://github.com/lkreidberg/batman): BSD 3-Clause License
- [`astropy`](https://www.astropy.org/): BSD 3-Clause License
- [`emcee`](https://github.com/dfm/emcee): MIT License
- [`corner`](https://github.com/dfm/corner.py): MIT License
- [`matplotlib`](https://matplotlib.org/): PSF-based License
- [`numpy`](https://numpy.org/): BSD 3-Clause License
- [`pandas`](https://pandas.pydata.org/): BSD 3-Clause License
- [`PyYAML`](https://pyyaml.org/): MIT License

These dependencies are not included in this repository and are used under their respective licenses.

---

## 🛰 License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).

You are free to use, modify, and distribute this software under the terms of the GPL. If you distribute modified versions, you must also release your source code under the same license.

See the `LICENSE` file for full details.

