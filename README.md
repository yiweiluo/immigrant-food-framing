# immigrant-food-framing
code and data for project on Othering and Low Prestige Framing of Immigrant Food in US Restaurant Reviews and LLMs

# Getting started
- Create and activate a Python 3.6.8 virtualenv (repo has not been tested with other versions)
- Install requirements: `pip install -r requirements.txt`
- Download spaCy models:
	- `python -m spacy download en_core_web_lg`
	- `python -m coreferee install en`
- Download [Yelp Open dataset](https://www.yelp.com/dataset) *NB: This dataset appears to be frequently updated by Yelp*
- Download synthetic GPT-3.5 reviews and other larger data files from [this drive folder](https://drive.google.com/drive/folders/1HkQxVasiLBcW-VNtpOapKZ_8xdP-mpdm?usp=sharing)
    - Place the file `gpt3_reviews_concat.csv` in a subdirectory with the path `data/llm` 
    - Unzip `spacy_processed_coref_output.zip` and place the contents in a subdirectoy with the path `data/yelp/restaurants_only`
