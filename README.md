# FastText Word Embeddings on Yelp Tips

A Natural Language Processing project comparing three FastText embedding setups using Yelp tips text:

1. Train FastText from scratch on Yelp text.
2. Inspect pretrained FastText crawl vectors.
3. Initialize Yelp vocabulary from pretrained vectors and adapt the model through further training on Yelp text.


## Requirements

Python 3.9+ is recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

## Datasets and pretrained vectors

This notebook uses two external resources that are **not included** in this repository:

- Yelp tips JSONL file: `yelp_academic_dataset_tip.json`
- Pretrained FastText crawl vectors: `crawl-300d-2M.vec`

Place them in your Kaggle input area or local data folders, then set `YELP_JSONL_PATH` and `PRETRAINED_VECTOR_PATH` near the top of the notebook if automatic file discovery does not find them.

The notebook reads up to 50,001 Yelp records to match the original experiment. The pretrained vector file is large; `PRETRAINED_LIMIT` defaults to 200,000 entries to reduce RAM usage. Increase it only if your machine has sufficient memory.

## Run

Open the notebook in Jupyter or Google Colab/Kaggle and run cells from top to bottom. The notebook preprocesses the text, trains the Yelp FastText model, loads pretrained vectors, adapts a FastText model to Yelp, compares nearest neighbors, and exports experiment outputs.

## Generated outputs

After a successful run, the `outputs/` folder contains:

- `nearest_neighbors_comparison.csv`
- `mean_similarity_comparison.csv`
- `experiment_summary.csv`
- `fasttext_yelp.model`
- `fasttext_yelp_adapted.model`


## Important interpretation note

The original notebook called results from a negated word vector “opposite words.” Such results are not guaranteed antonyms. This version labels them as *opposite-direction neighbors* in an optional exploratory cell and keeps the main comparison focused on nearest neighbors.

## Project scope

This is an embedding comparison experiment, not a sentiment classification model. Similarity scores and nearest-neighbor examples are descriptive and depend on corpus sampling, preprocessing, pretrained-vector coverage, and training settings.
