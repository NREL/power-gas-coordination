# power-gas-coordination
This serves as a repository for python code used to conduct an analysis of coordinated power system and gas network operations for the Colorado Front Range region. The included python scripts co-simulate gas operations (modeled using [SAInt](https://encoord.com)) with power system operations (modeled using [PLEXOS](https://energyexemplar.com/solutions/plexos/)), both of which are commercially available software tools. The primary information passed between the models is the amount of natural gas requested by gas-fired electric generators (PLEXOS -> SAInt) and the feasible quantity of natural gas that can be delivered to nodes in the gas network, including gas generators (SAInt -> PLEXOS). 

The modules includes the following scripts:

- classes.py - a collection of top-level object classes used throughout the code, including classes for model objects and PLEXOS/SAInt instatiations.
- market_function.py - defines the main function for running interleaved power and gas simulations.
- helper_functions.py - a collection of useful functions for setting up input files, running the model, and manipulating and saving data.
- scenarios.py - sets up the models for analysis (based on model structure created in classes.py).
- run_models.py - runs the models specified in scenarios.py (executed by the user).

Due to the use of proprietary and sensitive data in this analysis we are not able to publish the data accompanying this analysis. Despite this, we have published the codebase here to illustrate our framework. Please see the following publications for additionl details on the analysis:
- Guerra et al., "Coordinated Operation of Electricity and Natural Gas Systems from Day-ahead to Real-time Markets". Available at https://www.sciencedirect.com/science/article/pii/S0959652620348034?via%3Dihub. 
- Guerra et al., "Electric Power Grid and Natural Gas Network Operations and Coordination". NREL Technical report. Available at https://www.nrel.gov/docs/fy20osti/77096.pdf.

NREL is currently developing a suite of tools to enable co-simulation of gas and grid operations based on the HELICS co-simulation platform. Those interested in learning more about NREL's efforts in this space are encouraged to check out the [HELICS documentation](https://docs.helics.org/en/latest/) or [codebase on Github](https://github.com/GMLC-TDC/HELICS), or to contact the HELICS natural gas use case lead Brian Sergi (bsergi@nrel.gov) for more information.  

Conceptual diagram of the PLEXOS-SAInt coordination and data exchange:

[Coordination diagram - Paper figure.pdf](https://github.com/NREL/power-gas-coordination/files/8111995/Coordination.diagram.-.Paper.figure.pdf)
