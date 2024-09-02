# Predicting Hotspot Numbers in Kalimantan Using Gaussian Process Regression
This repository contains the implementation of a model to predict the number of hotspots in Kalimantan, Indonesia, using Gaussian Process Regression (GPR) based on climate indicators. The research was conducted as part of a thesis at the Department of Mathematics, Faculty of Mathematics and Natural Sciences, Institut Pertanian Bogor.

## Overview

Forest fires in Kalimantan are a recurring natural disaster influenced heavily by both local and global climate indicators. This project develops a predictive model with low hyperparameter complexity, leveraging the Gaussian Process Regression method to estimate the number of hotspots based on various climate indicators.

### The key objectives of this study are:

- To construct a Gaussian Process Regression model using climate data such as rainfall, days without rain, and global climate indicators like El Niño-Southern Oscillation (ENSO) and Indian Ocean Dipole (IOD).
- To optimize the model using three hyperparameter tuning methods: Bayesian optimization, grid search, and random search.

### Key Features

- Data: The model uses data collected from various sources, including local climate data (rainfall, rainfall anomaly, days without rain) and global climate indicators (ENSO, IOD).
- Modeling: Gaussian Process Regression (GPR) with Automatic Relevance Determination (ARD) is used to model the relationship between climate indicators and the number of hotspots.
- Kernel Functions: The ARD Squared Exponential kernel is primarily used due to its ability to handle different scales of input features effectively.
- Hyperparameter Tuning: The model's performance is optimized using Bayesian optimization, grid search, and random search techniques, ensuring the best fit for both training and testing datasets.

### Results

The best-performing model was achieved using Bayesian optimization and random search for hyperparameter tuning. The model achieved the following accuracy metrics on the test set:

Bayesian optimization
- RMSE: 844.47
- MAE: 354.29
- R-squared: 54.52%

random search
- RMSE: 846.58
- MAE: 350.93
- R-squared: 54.29%

These results indicate a good fit for predicting the number of hotspots based on the climate indicators provided.

## License

This work is copyrighted by Institut Pertanian Bogor (IPB) as of 2024. All rights are protected under applicable laws.

### Usage Restrictions:

- You are prohibited from citing parts or the entirety of this work without properly acknowledging the source. Citations must be for the purposes of education, research, scientific writing, report preparation, criticism, or review of a specific issue, and must not harm the interests of IPB.
- You are prohibited from publishing or reproducing parts or the entirety of this work in any form without explicit permission from IPB.

For any usage outside these terms, please contact Institut Pertanian Bogor for permission.

## Acknowledgements

This research was conducted under the supervision of Prof. Dr. Ir. Sri Nurdiati, M.Sc., and Mochamad Tito Julianto, M.Kom. Special thanks to the Institut Pertanian Bogor for their support and resources.
