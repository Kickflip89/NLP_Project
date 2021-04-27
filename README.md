# NLP_Project
Code repository for a short-story generator based on https://arxiv.org/abs/1805.04833

Some methods were adapted from https://github.com/pytorch/fairseq/tree/master/fairseq
under the MIT license and are not being used for profit.

Model works with the WritingPrompts dataset: https://www.kaggle.com/ratthachat/writing-prompts

## ./utils
contains the baseline single and multi-head attention classes and functions
in Attention.py

## ./embedding
* loader.py - DataSet class to stream batches from disk
* trained.mdl - original w2v model (300 dimensions)
* w2v_128.mdl - w2v model used (128 dimensions)

## ./data
contains some raw data from the last 5800 iterations
of the convolutional and linear versions of the model

## ./notebooks
Jupyter notebooks containing some of the actual experiments run

## ./
* model.py - wrapper class for encoder-decoder architecture
* train.py - example training driver
