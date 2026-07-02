rule all:
    input:
        "outputs/figures/results.pdf",
        "outputs/figures/pinn_surface.html",
        "outputs/models/pinn.pt"


rule derive_symbolic:
    output:
        "outputs/symbolic_done.txt"
    shell:
        """
        uv run python src/symbolic_derivations.py
        """


rule ingest_and_vectorize:
    input:
        "outputs/symbolic_done.txt"
    output:
        "outputs/vectorization_done.txt"
    shell:
        """
        uv run python src/numerical_core.py
        """


rule analyze_stability:
    input:
        "outputs/vectorization_done.txt"
    output:
        "outputs/stability_done.txt"
    shell:
        """
        uv run python src/analyse_stabilite.py
        """


rule train_pinn:
    input:
        "outputs/stability_done.txt"
    output:
        "outputs/models/pinn.pt"
    shell:
        """
        uv run python src/deep_pinn.py
        """


rule generate_plots:
    input:
        "outputs/models/pinn.pt"
    output:
        "outputs/figures/results.pdf",
        "outputs/figures/pinn_surface.html"
    shell:
        """
        uv run python src/visualization.py
        """