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
```
phd_integrator_project
├── .coverage
├── .git
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
│       └── .gitkeep
├── outputs
│   ├── figures
│   │   ├── pinn_surface.html
│   │   └── results.pdf
│   ├── models
│   │   └── pinn.pt
│   ├── stability_done.txt
│   ├── symbolic_done.txt
│   └── vectorization_done.txt
├── pyproject.toml
├── questions_de_reflexion.md
├── src
│   ├── __init__.py
│   ├── __pycache__
│   ├── analyse_stabilite.py
│   ├── deep_pinn.py
│   ├── hpc_acceleration.py
│   ├── numerical_core.py
│   ├── symbolic_derivations.py
│   └── visualization.py
├── tests
│   ├── __pycache__
│   ├── conftest.py
│   ├── test_hpc.py
│   ├── test_numerical.py
│   ├── test_pinn.py
│   ├── test_stability.py
│   ├── test_symbolic.py
│   └── test_visualization.py
└── uv.lock

```