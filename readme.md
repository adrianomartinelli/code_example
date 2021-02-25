# Install environment
clone repository and install conda environment
```bash
git clone https://github.com/adrianomartinelli/code_example.git
cd code_example
conda env create -f example_code_env.yml && conda activate code_example
```

# Run
Either interactively or in the shell with
```bash
python minst_example.py
```
Two plots will be produced:
- examples.png: examples of reconstructed images
- loss.png: Traning/validation and test loss development of the training

```bash
python ripleysK.py
```
A plot will be produced:
- ripleysK.png: RipleysK estimate for random data and test for deviation from CSR.



# File description
cellAutoencoder.py: Autoencoder model

trainer.py: General class to train models

minst_example.py: Example on the MINST data set

ripleysK.py: Base implementation of Ripley's K. To be extended with correction methods.

graph_builder module: module to construct graphs from cellmasks. No running example available, only examples on how us the module in `example.py` file.
