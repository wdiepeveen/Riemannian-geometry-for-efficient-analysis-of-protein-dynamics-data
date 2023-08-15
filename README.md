# Riemannian geometry for efficient analysis of protein dynamics data

    [1] W. Diepeveen, C. Esteve-Yagüe, J. Lellmann, O. Öktem, C-B. Schönlieb,  
    Riemannian geometry for efficient analysis of protein dynamics data
    arXiv preprint arXiv:----.-----. 2023 Aug -.

Setup
-----

The recommended (and tested) setup is based on MacOS 13.4.1 running Python 3.8. Install the following dependencies with anaconda:

    # Create conda environment
    conda create --name rieprogeo1 python=3.8
    conda activate rieprogeo1

    # Clone source code and install
    git clone https://github.com/wdiepeveen/Riemannian-geometry-for-efficient-analysis-of-protein-dynamics-data.git
    cd "Riemannian-geometry-for-efficient-analysis-of-protein-dynamics-data"
    pip install -r requirements.txt


Reproducing the experiments in [1]
----------------------------------

The jupyter notebook `experiment/protein_conformation_processing.ipynb` has been used to produce the results in [1]. 
* For the adenylate kinase results use `struct = 1`.
* For the SARS-CoV-2 helicase nsp 13 use `struct = 2`.
