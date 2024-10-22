# Under Development
This tool is currently under active development and will be released with full functionality, including example test cases, in January 2025.

# QBtoWEIS

**QBtoWEIS** is an extension of the WEIS (Wind Energy with Integrated Servo-control) framework that integrates the capabilities of QBlade to expand the design and optimization process for floating offshore wind turbines. By adding computationally highly optimized methods for wake aerodynamics and structural modeling, QBtoWEIS creates an even more versatile and powerful multi-fidelity toolchain, offering greater flexibility in co-design and optimization.

## WEIS

[WEIS](https://github.com/WISDEM/WEIS) is a comprehensive framework developed by the National Renewable Energy Laboratory (NREL) that integrates several NREL-developed tools for the design optimization of floating offshore wind turbines. It facilitates aero-servo-hydro-elastic modeling, structural analysis, and control system design, enabling a holistic approach to floating wind turbine development.

## QBlade

[QBlade](https://qblade.org/) is a versatile aero-servo-hydro-elastic simulation tool for wind turbines. It features a highly optimized lifting-line free vortex wake algorithm, which ensures efficient and accurate wake modeling. The underlying structural model implemented in QBlade supports Euler-Bernoulli, Timoshenko, and geometrically exact beam theory, all of which provide computationally efficient structural models. Notably, all three approaches can be associated with similar computational costs, allowing for flexibility in model selection without sacrificing performance.

[QBlade documentation](https://docs.qblade.org/)

## Additional Packages

QBtoWEIS integrates the following packages in addition to the stack of tools already available in WEIS:
* [QBlade](https://qblade.org/) - freely available wind turbine simulation tool

## Additional Packages (coming soon)

* [SONATA](https://github.com/ptrbortolotti/SONATA) - toolbox for Multidisciplinary Rotor Blade Design Environment for Structural Optimization and Aeroelastic Analysis

## Installation

The installation process is almost equivalent to the one of the main branch of WEIS:

On laptop and personal computers, installation with [Anaconda](https://www.anaconda.com) is the recommended approach because of the ability to create self-contained environments suitable for testing and analysis. WEIS requires [Anaconda 64-bit](https://www.anaconda.com/distribution/). However, the `conda` command has begun to show its age, and we now recommend the one-for-one replacement with the [Miniforge3 distribution](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3), which is much more lightweight and more easily solves for the package dependencies. Sometimes, using `mamba` in place of `conda` with this distribution speeds up the installation process. QBtoWEIS is currently supported on Linux and Windows Subsystem for Linux (WSL2). Installing QBtoWEIS on native Windows is not yet supported, but planned in 2025.

The installation instructions below use the environment name, "qbweis-env," but any name is acceptable. For those working behind company firewalls, you may have to change the conda authentication with `conda config --set ssl_verify no`. Proxy servers can also be set with `conda config --set proxy_servers.http http://id:pw@address:port` and `conda config --set proxy_servers.https https://id:pw@address:port`.

1. Clone the repository and create a virtual environment install the software:
   
        conda config --add channels conda-forge
        git clone https://github.com/rbehrensdeluna/QBtoWEIS.git
        cd QBtoWEIS

        
2. Create a virtual environment abd install the software
   
        conda env create --name qbweis-env -f environment.yml
        conda activate qbweis-env   # (if this does not work, try source activate qbweis-env)
        conda install -y petsc4py mpi4py 
        pip install --no-deps -e . -v
        conda install -c conda-forge pyoptsparse

**NOTE:** To use QBtoWEIS again after installation is complete, you will always need to activate the conda environment first with conda activate qbweis-env (or source activate qbweis-env).

## Download and configure QBladeCE executables under WSL2

   Download QBladeCE from https://qblade.org/downloads/ and place it in the WSL2 directory.
   
   install some libraries that are required to run qblade:
       	
        sudo apt-get update -y
        sudo apt-get install -y libqt5opengl5 libqt5xml5 libquadmath0 libglu1-mesa
        
   Make QBladeCE executable
   
        chmod +x run_qblade.sh
        chmod +x QBladeCE_x.y.z # x.y.z should be replaced by the actual version number, e.g. 2.0.7.8 

**NOTE:** QBtoWEIS requires QBladeCE/QBladeEE version 2.0.7.8 or newer to enable for multi-processing capabilities (number_of_workers > 1)
        
## Troubleshoot.
If you are having trouble creating the virtual environment try allocating more RAM to the WSL2 (e.g. https://learn.microsoft.com/en-us/answers/questions/1296124/how-to-increase-memory-and-cpu-limits-for-wsl2-win)

Verify if correct package versions were installed

        conda list wisdem # (check if wisdem version equals 3.13.0 otherwise uninstall and revert)
        pip uninstall wisdem
        pip install wisdem==3.13.0
        
        The same should be done for rosco (2.9.2) scipy (1.13.0) and openfast (3.5.2)
