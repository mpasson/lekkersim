[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# LekkerSIM package

LekkerSIM is a open source package for linear simulation of photonic circuit, based on the well known scattering matrix method.
Its features include:
- Easy calculation of S-parameters of a photonic circuit;
- A collection of pre-defined building blocks for easy definition of circuits;
- Parametric building blocks;
- Hierarchical circuit definition;
- Monitors inside the circuit for calculation of power flow.
- Loading and exporting scattering matrices in [InPulse](https://cordis.europa.eu/project/id/824980) data format.

## Installation
After cloning, just add the repository folder to the PYTHONPATH. 

Requirements are in the attached [requirements.txt](requirements.txt) file. They can be installed by running:
    
    pip install -r requirements.txt

## Documentation
Full documentation is available on [Read the Docs](https://lekkersim.readthedocs.io)

### Building documentation yourself
You can build your own local documentation by running the command [localdocs](localdocs). In this case, you need the following extra packages:
 - Python packages:
   - ```sphinx```
   - ```sphinx-rtd-theme```
   - ```nbsphinx```
   - ```notebook```
 - System packages:
   - ```pandoc```


