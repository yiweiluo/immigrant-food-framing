# Data preprocessing

Once you have completed the getting started steps, the scripts in this directory can be used to filter and preprocess Yelp and synthetic GPT reviews. 

`0_filter_restaurant_data.py` ingests the full dataset from the `--path_to_data_dir` argument, finds all businesses with the categories of either "restaurants" or "food," and saves that subset of the business data and a subset of the reviews associated with those restaurants to `--out_dir`. This step is relevant only for the Yelp review data.

`1_spacy_process_texts.py` processes the raw review texts in batches with spaCy and the Coreferee add-on for coreference resolution. GPU recommended for efficient processing. All default arguments can be used to process Yelp reviews. To process GPT reviews, run the following:
```
python 1_spacy_process_texts.py --path_to_texts ../data/llm/gpt3_reviews_concat.csv --out_dir ../data/llm/spacy_processed --text_fields review_no_disclaimers
```

`2_extract_frames.py` extracts adjectival features from the parsed review data and creates a lookup dictionary `frames_lookup.pkl` that stores the extracted features per review. All default arguments can be used to process Yelp reviews. To process GPT reviews, run the following:
```
python 2_extract_frames.py --path_to_raw ../data/llm/gpt3_reviews_concat.csv --path_to_parsed ../data/llm/spacy_processed --out_dir ../data/llm --guid lookup_guid --text_field review_no_disclaimers
```
