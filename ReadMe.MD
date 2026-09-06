# Gesture Validation Toolkit for Dynamic Time Warping

There is a rapidly growing interest in quantifying gesture form similarity using Dynamic time warping (DTW) in the field of (psycho)linguistics (example studies: [Pouw & Dixon, 2020](https://doi.org/10.1080/0163853X.2019.1678967); [Akamine, Meyer, & Ozyurek, 2026](https://osf.io/preprints/psyarxiv/kjhd2_v1/); [Pouw, de Wit, et al., 2021](https://doi.org/10.1007/978-3-030-77817-0_20); [Pouw, Dingemanse, Motamedi, & Özyürek, 2021](https://doi.org/10.1111/cogs.13014); [Trujillo, Dideriksen, Tylén, Christiansen, & Fusaroli, 2023](https://doi.org/10.1111/cogs.13298)).

Recently, [Akamine, Dingemanse, & Özyürek (2026)](https://doi.org/10.3758/s13428-026-02975-5) validated DTW against human similarity ratings and found a moderate negative correlation between hand-coded simiarity ratings and DTW distance scores, confirming that DTW offers a continuous, scalable measure of gesture form similarity (see also [Pouw, Dingemanse, et al., 2021](https://doi.org/10.1111/cogs.13014); [Pearson, Nuttall, & Pouw, 2025](https://doi.org/10.1111/cogs.70137) for other validations that were less extensive).

Although [Akamine et al.’s (2026)](https://doi.org/10.3758/s13428-026-02975-5) validation work demonstrated the utility of DTW, they did not compare the performance of different variants of DTW (e.g., shapeDTW). This module extends the previous validation work by providing an easy-to-use toolkit that allows users to compare the performance of different DTW variants or other bivariate time series (see e.g., [Cliff, Bryant, Lizier, Tsuchiya, & Fulcher, 2023](https://doi.org/10.1038/s43588-023-00519-x)).

We aim to provide a toolkit for kinematic similarity analyses of communicative actions that will help discover new time series analyses tailored to this domain.


## Citation

If you found this module helpful, please cite it as below:

Akamine, S., Samson, K., & Pouw, W. (2026). Gesture validation toolkit for dynamic time warping: Benchmarking DTW variants against human judgements of gesture form similarity (Version 1.0.0) [Computer software]. https://github.com/WimPouw/envisionboxDTW/

```
@software{Akamine_GestureValidationToolkit_2026,
author = {Akamine, Sho; Samson, Kyle; Pouw, Wim},
month = sep,
title = {{Gesture validation toolkit for dynamic time warping}},
url = {},
version = {1.0.0},
year = {2026}
}
```

## Requirements
Before running the script, you need to have the following installed:
- [Miniconda](https://www.anaconda.com/download/success)
- [Visual Studio Code](https://code.visualstudio.com/download) (or an IDE of your preference).


## Instructions
You can run the module by following the steps below:

1. Download the repository. You can do so in two different ways:
    - Option 1: clone the repository
    - Option 2: click the green "<> Code" button on this page and select "Download ZIP". Make sure to unzip the folder before moving on to the next step. 

2. Set up the conda environment by following the instructions below:

  1. Open terminal/anaconda prompt at the "scripts" folder
          - Mac:
              1. go to the `envisionboxDTW` or `envisionboxDTW-main` folder
              1. right-click the "scripts" folder
              1. select "open terminal at this folder" (under "Services") <br><br>
          - Windows:
              1. go to the "scripts" folder
              1. copy the path to the folder
              1. open Anaconda Prompt
              1. type cd and paste the path after a space (e.g., cd D:/users/shoakamine/envisionboxDTW/scripts)
              1. if the folder is not in the C drive (e.g., D:/users/...), type the drive initial followed by a colon (e.g., D:):
    
      1. Create a Python 3.10 environment in the Anaconda prompt/terminal
          ```
          conda create --name dtw_validation python=3.10
          ```
      1. Activate your conda environment
          ```
          conda activate dtw_validation
          ```
      
      1. Install git if not already installed: `conda install -c anaconda git`
      
      1. Run this command to install required packages: `pip install -r requirements.txt`
      
3. Open the `dtw_validation.ipynb` file

4. In VS Code, click "select kernel" on the top right and select "Python Environments..." --> "dtw_validation"
