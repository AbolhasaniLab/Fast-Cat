This software runs the FastCat control algorithm for either Maximum Variance or qNEHVI campaigns, directed towards user selected linear or branched product. The algorithm is pretrained on real-world collected data sets that are included in the "FastCat Control" folder (HFNonDimInputs.csv and HFNonDimOutputs.csv), and it modifies an active input file (FlowConditions.csv) to provide the hardware control the next test conditions. The algorithm will then wait until the data output folder (GC Output Files) is updated to reflect the result of the previous condition, then the data is processed and a new condition is suggested. During regular system operation, the GC Output Files folder is automatically updated by the analysis software; however, it may be updated manually for testing purposes (copy TestOutput.txt to FastCat Control/GC Output Files).

Instructions for running program:
1. Open the files "HFNonDimInputs.csv" and "HFNonDimOutputs.csv", and check that the values are initialized to at least 4 prior conditions. Supplied Files contain initialization, normal, and branched aldehyde optimization run conditions for L1. 

The GC Output Files folder should contain only:
"TestOutput.txt"

The FlowConditions.csv file should contain only:
"N2 mLn/min,CO mLn/min,H2 mLn/min,Mix mLn/min,T C,Octene Feed,Ligand Feed,Rh Feed,Solvent Feed,Collection,Pressure,Eq Time,Rxn Time,Wash,Replicate" as a header

All .csv files should be closed when running the script otherwise python will not be able to append new conditions or results during processing.

2. Run the primary script ("FastCat Code/FastCat Loop/Ensemble_Loop.py")

3. Select if the run is a test run. Console:"Test Run? (y/[n]):" Default is no. This option will reduce the run time for testing by lowering the ensemble size and batch/sampling size for monte carlo sampling.

4. Select directory containing control files and prior data (FastCat Code/FastCat Control)

5. Select directory containing data outputs (FastCat Code/FastCat Control/GC Output Files)

6. Choose Bayesian Optimization acquisition function. Console:"Choose Selection Policy ([1: MV], 2: qNEHVI):". Default is MV

7. Choose target aldehyde product. Console:"Choose Optimization Target ([1: Linear], 0: Branched):" Defualt is Linear aldehyde

8. Once the optimization loop is finished, 2 new conditions should be appended to both the "HFNonDimInputs.csv" and "FlowConditions.csv" list and the code is wating for the physical system control to complete the first condition.

9. To simulate a sample, copy the "TestOutput.txt" to the "GC Output Files" directory which should append the new result to "HFNonDimOutputs.csv" and inititate a new optimization cycle with 1 new condition being appended to "HFNonDimInputs.csv" and "FlowConditions.csv"

The loop will generate a new condition from the entire dataset for every new file that shows up in the GC output file directory.

Normal run will take a few minutes per optimizaion cycle

System Specifications:
Windows 10 Education 10.0.19044 Build 19044
Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz, 3000 Mhz, 18 Core(s), 36 Logical Processor(s)
Installed Physical Memory (RAM)	64.0 GB
NVIDIA RTX A5000 GPU

Environment Specifications:
FastCat_Loop_Environment.yml - Conda Environment File


