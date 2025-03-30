# Multi-Task Learning

This project implements a sentence transformer model and expands it into a multi-task learning (MTL) framework with two tasks:
- Task A: Sentence Classification (11 emotion labels)
- Task B: Sentiment Analysis (positive, neutral, negative)

## Dataset

We use the [TweetEval](https://huggingface.co/datasets/tweet_eval) benchmark dataset:
- `tweet_eval/emotion` -> Task A
- `tweet_eval/sentiment` -> Task B

## Project Structure

- `src/train.py`: Training logic
- `src/data.py`: Multi-task dataset loader, util functions to get dataset and embeddings
- `src/model.py`: Multi-taks model definition
- `notebooks/tasks.ipynb`: Notebook to test overfitting on a small batch

## Training

```bash
pip install -r requirements.txt
python src/train.py
```

## Docker

Build and run using Docker:

```bash
docker build -t mtl .
docker run -it mtl
```
