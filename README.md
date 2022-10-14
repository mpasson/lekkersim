# GenSol package

GenSol is a open source package for linear simulation of photonic circuit, based on the well known scattering matrix method.
Its features include:
- Easy calculation of S-parameters of a photonic circuit;
- A collection of pre-defined building blocks for easy definition of circuits;
- Parametric building blocks;
- Hierarchical circuit definition;
- Monitors inside the circuit for calculation of power flow.
- Loading and exporting scattering matrices in [InPulse](https://cordis.europa.eu/project/id/824980) data format.

## Installation
After cloning, just add the repository folder to the PYTHONPATH. 

Requirements are in the attached [requirements.txt](requirements.txt) file. They can be installed by runnig:
    
    pip install -r requirements.txt

## Documentation
The full documentation of the software is available [Here](https://marco_passoni.gitlab.io/gensol/index.html)

### Building documentation yourself
You can build your own local documentation by running the command [localdocs](localdocs). In this case, you need the following extra packages:
 - Python packages:
   - ```sphinx```
   - ```sphinx-rtd-theme```
   - ```nbsphinx```
   - ```notebook```
 - System packages:
   - ```pandoc```


