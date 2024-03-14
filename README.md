# immigrant-food-framing
Code and data for [paper](https://arxiv.org/abs/2307.07645) on Othering and Low Prestige Framing of Immigrant Food in US Restaurant Reviews and LLMs.

# Getting started
- Create and activate a Python 3.6.8 virtualenv (repo has not been tested with other versions)
- Install requirements: `pip install -r requirements.txt`
- Download spaCy models:
	- `python -m spacy download en_core_web_lg`
	- `python -m coreferee install en`
- Download [Yelp Open dataset](https://www.yelp.com/dataset) *NB: This dataset appears to be frequently updated by Yelp*
- Download synthetic GPT-3.5 reviews and other larger data files from [this drive folder](https://drive.google.com/file/d/1-6pzufs7e-z-IM-PucOHiOTFS6zloouI/view?usp=sharing) if you do not want to create them from scratch. Then, place the file `gpt_new_reviews.csv` in a subdirectory with the path `data/llm`.
