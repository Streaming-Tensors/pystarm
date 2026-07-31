# Compiling and running pystarm on Argonne's Bebop Systems

Author: Stefan Smith
June 2026

## Doing this via shell script

cd into the scripts directory, and run

```
chmod +x bebop-setup.sh
```

then run it using

```
./bebop-setup.sh
```

## Environment Dependencies

You will need to import anaconda using:

```
module load miniforge
```

Once that's done, you can now load or create your own conda environment. You can create a conda environment like this:

```
conda init
conda create -n pystarm-env
```

You will need these python dependencies:
- Numpy
- pybind11
- pyttb
- MKL

To install those on your conda environment:

```
conda install pip
pip install Numpy pybind11 pyttb MKL
```

Now load the rest of the necessary modules:

```
module load gcc
module load intel-oneapi-mkl
```

## Compiling

In order to compile, we need to make sure make uses the correct compiler. On bebop, run this:

```
CC=g++ make
```

and everything should compile. You will get a lot of warnings, but you can ignore those. THIS COMMAND IS CASE SENSITIVE. Please make sure the CC is capital! You will get a linker error of some sort if you use `cc=g++ make`.

## Running

To make sure everything runs, simply run

```
python3 test.py
```

## FOR SYSTEMS WITHOUT MKL in MODULE

You probably installed MKL via pip in a conda environment. To get this to compile you need to create a symlink, after activating your conda environment:

```
mkdir -p $CONDA_PREFIX/lib/intel64
ln -s $CONDA_PREFIX/lib/libmkl*.a $CONDA_PREFIX/lib/intel64/
```

After that, you should be able to run this: `export MKLROOT=$CONDA_PREFIX`

## Requesting an interactive node on bebop:

```
qsub -q bdwall -l select=1:ncpus=36:mpiprocs=36 -l walltime=30:00 -A StarM-HPC-Updated -I
```