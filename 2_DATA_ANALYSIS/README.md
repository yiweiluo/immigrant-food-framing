# Review analysis

Once you have completed the data preprocessing, the scripts in this directory can be used to aggregate extracted features for the framing and regression analysis.

`0_aggregate_frames_data.py` reads in the feature lookup dictionary created during preprocessing and aggregates individual features into quantitative scores along broader framing dimensions using the feature dictionaries within `feature_dicts/`. 

`1_prep_regression_data.py` creates a dataframe with one row per review containing all review variables (framing scores, length, restaurant and neighborhood attributes, etc.) for the regression analyses.

`2_regressions.py` performs regressions and saves the results. 

`3_fightin_words.py` computes z-scores of weighted log odds ratios to qualitatively analyze features associated with each cuisine (region) using the method from Monroe et al. (2008).

`plotting.ipynb` is a Jupyter notebook for visualizing results.