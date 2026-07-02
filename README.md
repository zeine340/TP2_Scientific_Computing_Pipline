[![Scientific Computing CI/CD Pipeline](https://github.com/zeine340/TP2_Scientific_Computing_Pipline/actions/workflows/ci_cd_pipeline.yml/badge.svg)](https://github.com/zeine340/TP2_Scientific_Computing_Pipline/actions/workflows/ci_cd_pipeline.yml)

## Project overview

Physics-Informed Neural Network for solving the Advection-Diffusion equation.

## Environment

```
uv sync
```

or

```
uv sync --locked
```

## Run the complete workflow
```
snakemake --cores all
```
## Run the tests
```
uv run pytest
```
or with coverage:
```
uv run pytest --cov=src --cov-report=term-missing
```
## Reproducibility
The project is fully reproducible using

```
uv sync

snakemake --cores all
```

## Project structure
phd_integrator_project
├── .github
│   └── workflows
│       └── ci_cd_pipeline.yml
├── .gitignore
├── .venv
├── README.md
├── Snakefile
├── data
│   ├── .gitkeep
│   └── raw_sensors
├── outputs
│   ├── figures
│   └── models
├── pyproject.toml
├── questions_de_reflexion(suite).md
├── questions_de_reflexion.md
├── src
│   ├── __init__.py
│   ├── analyse_stabilite.py
│   ├── deep_pinn.py
│   ├── hpc_acceleration.py
│   ├── numerical_core.py
│   ├── symbolic_derivations.py
│   └── visualization.py
├── tests
│   ├── conftest.py
│   ├── test_symbolic.py
│   ├── test_numerical.py
│   ├── test_stability.py
│   ├── test_pinn.py
│   └── test_visualization.py
└── uv.lock

