# Review analysis

Once you have completed the data preprocessing, the scripts in this directory can be used to aggregate extracted features for the framing and regression analysis.

`0_aggregate_frames_data.py` reads in the feature lookup dictionary created during preprocessing and aggregates individual features into quantitative scores along broader framing dimensions using the feature dictionaries within `feature_dicts/`. For Yelp reviews, all default command line arguments can be used. For GPT reviews, run the following:
```
python 0_aggregate_frames.py --path_to_lookup ../data/llm/frames_lookup.pkl --out_dir ../data/llm
```

`1_prep_regression_data.py` creates a dataframe with one row per review containing all review variables (framing scores, length, restaurant and neighborhood attributes, etc.) for the regression analyses. For Yelp reviews, run the following:
```
python 1_prep_regression_data.py --do_yelp
```
For GPT reviews, run the following:
```
python 1_prep_regression_data.py --path_to_raw_reviews ../data/llm/gpt_new_reviews.csv --path_to_framing_scores ../data/llm/agg_frame_lookups --guid guid --text_field review_no_disclaimers --out_dir ../data/llm 
```

`2_regressions.py` performs regressions on reviews and saves the results to `/results/`. For Yelp reviews, run the following:
```
python 2_regressions.py --do_yelp
```
For GPT reviews, run the following:
```
python 2_regressions.py --path_to_reviews_df ../data/llm/per_reviews_df.csv --out_dir llm_results
```
You can also optionally specify the `--model` and `--prompt_index` arguments to subset the analyses to reviews generated using those parameters.

`3_fightin_words.py` computes z-scores of weighted log odds ratios to qualitatively analyze features associated with each cuisine (region) using the method from Monroe et al. (2008). For Yelp reviews, run the following:
```
python 3_fightin_words.py --do_yelp
```
For GPT reviews, run the following:
```
python 3_fightin_words.py --path_to_reviews_df ../data/llm/per_reviews_df.csv --path_to_lookup ../data/llm/frames_lookup.pkl --out_dir llm_fightin_words_results
```
Again, you can optionally specify the `--model` and `--prompt_index` arguments to subset the analyses to reviews generated using those parameters.

Finally, `plotting.ipynb` is a Jupyter notebook for visualizing regression results.