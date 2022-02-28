# Combining graph neural networks and classical kernels for graph generation
Pytorch implement of the article Combining graph neural networks and classical kernels for graph generation.

This library refers to the source code https://github.com/yongqyu/MolGAN-pytorch

## Dependencies
1) python>=3.5
2) pytroch>=1.10.0  https://pytorch.org
3) rdkit  https://www.rdkit.org/
4) numpy

##Setup
Before running:
1) Enter the data folder and run download_dataset.sh or extract the files at gdb9.tar.gz in the data folder
2) From the data folder run:
>python sparse_molecular_dataset.py 

##Run the code
>python main.py --name runname
Will result with the run's logs at /results/runname/logs.txt
Samples at /results/runname/fake_samples
Models at /results/runname/models




