# Data preprocessing

Once you have completed the getting started steps, the scripts in this directory can be used to filter and preprocess Yelp and synthetic GPT reviews. 

`0_filter_restaurant_data.py` ingests the full dataset from the `--path_to_data_dir` argument, finds all businesses with the categories of either "restaurants" or "food," and saves that subset of the business data and a subset of the reviews associated with those restaurants to `--out_dir`. This step is relevant only for the Yelp review data.

`3_spacy_process_texts.py` processes the raw review texts in batches with spaCy and the Coreferee add-on for coreference resolution. GPU recommended for efficient processing. All default arguments can be used to process Yelp reviews. To process GPT reviews, run the following:
```
python 3_spacy_process_texts.py --path_to_texts ../data/llm/gpt_new_reviews.csv --out_dir ../data/llm/spacy_processed --text_fields review_no_disclaimers
```

`4_extract_frames.py` extracts adjectival features from the parsed review data and creates a lookup dictionary `frames_lookup.pkl` that stores the extracted features per review. All default arguments can be used to process Yelp reviews. To process GPT reviews, run the following:
```
python 4_extract_frames.py --path_to_raw ../data/llm/gpt_new_reviews.csv --path_to_parsed ../data/llm/spacy_processed --out_dir ../data/llm --guid guid --text_field review_no_disclaimers
```

If you wish, you can also run `1_get_synthetic_reviews.py` to collect your own GPT reviews (instead of using our dataset), though you will need an OpenAPI key and credits. Usage is as follows:
```
python 1_get_synthetic_reviews.py --api_key <ADD YOUR API KEY HERE> 
```

Once you have collected GPT reviews, run `2_aggregate_synthetic_reviews.py` to consolidate them into a single dataframe. You can use all default parameters.