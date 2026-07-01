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
        python src/symbolic_derivation.py
        touch {output}
        """


rule ingest_and_vectorize:
    input:
        "outputs/symbolic_done.txt"
    output:
        "outputs/vectorization_done.txt"
    shell:
        """
        python src/numerical_core.py
        touch {output}
        """


rule analyze_stability:
    input:
        "outputs/vectorization_done.txt"
    output:
        "outputs/stability_done.txt"
    shell:
        """
        python src/stability_analysis.py
        touch {output}
        """


rule train_pinn:
    input:
        "outputs/stability_done.txt"
    output:
        "outputs/models/pinn.pt"
    shell:
        """
        python src/deep_pinn.py
        """


rule generate_plots:
    input:
        "outputs/models/pinn.pt"
    output:
        "outputs/figures/results.pdf",
        "outputs/figures/pinn_surface.html"
    shell:
        """
        python src/visualization.py
        """