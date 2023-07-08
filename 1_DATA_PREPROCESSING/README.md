# Data preprocessing

Once you have completed the getting started steps, the scripts in this directory can be used to filter and preprocess Yelp businesses and reviews. 

`0_filter_restaurant_data.py` ingests the full dataset from the `--path_to_data_dir` argument, finds all businesses with the categories of either "restaurants" or "food," and saves that subset of the business data and a subset of the reviews associated with those restaurants to `--out_dir`. 

`1_spacy_process_texts.py` processes the raw restaurant review texts in batches with spaCy and the Coreferee add-on for coreference resolution. GPU recommended for efficient processing.

`2_extract_frames.py` extracts adjectival features from the parsed restaurant review data and creates a lookup dictionary `frames_lookup.pkl` that stores the extracted features per review. 
