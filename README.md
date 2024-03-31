# Text Analytics

## Language Moddeling

We create bigram, trigram and linear interpolation language models which are used for language generation and spell correction. 

[Source code](https://github.com/dimits-exe/text_analytics/tree/master/language_modeling) [Report](https://github.com/dimits-exe/text_analytics/blob/master/language_modeling/report.pdf)

## Sentiment Classification and POS Tagging tasks

We create deep learning models using the `Transformers\Datasets`, `Pytorch` and `Tensorflow` libraries. 
We also use the `keras_tuner` / `transformers_trainer` frameworks to optimize hyperparameters and model architecture.

We briefly mention additional tasks carried out:

* Sentiment Analysis: Dataset selection, exploratory analysis, custom stopwords, data augmentation. <!--- βαλε εδω αν εχω ξεχασει κατι -->
* POS Taggging: Dataset selection, exploratory analysis, custom parsing, custom baseline ("smart dummy") model, local caching of heavy computations, automated results generation (python -> LaTeX).

Each task features two IPython notebooks containing the executed code, python source files for repeated custom tasks and a unified report.

The reports discuss in detail the design decisions for each classifier and include graphs and aggregated results comparing the current model to the previous models.   

### Simple MLP model

[Sentiment classification](https://github.com/dimits-exe/text_analytics/blob/master/mlp_nlp/ex_9.ipynb) [POS Tagging](https://github.com/dimits-exe/text_analytics/blob/master/mlp_nlp/ex_10.ipynb) [Report](https://github.com/dimits-exe/text_analytics/blob/master/mlp_nlp/report.pdf)

### RNN Model

[Sentiment classification](https://github.com/dimits-exe/text_analytics/blob/master/rnn/ex_1.ipynb) [POS Tagging](https://github.com/dimits-exe/text_analytics/blob/master/rnn/ex2.ipynb) [Report](https://github.com/dimits-exe/text_analytics/blob/master/rnn/report.pdf)

### CNN Model

[Sentiment classification](https://github.com/dimits-exe/text_analytics/blob/master/cnn/ex_2.ipynb) [POS Tagging](https://github.com/dimits-exe/text_analytics/blob/master/cnn/ex3.ipynb) [Report](https://github.com/dimits-exe/text_analytics/blob/master/cnn/report.pdf)

### BERT Model

[Sentiment classification](https://github.com/dimits-exe/text_analytics/blob/master/transformers/ex_1.ipynb) [POS Tagging](https://github.com/dimits-exe/text_analytics/blob/master/transformers/ex3.ipynb) [Report](https://github.com/dimits-exe/text_analytics/blob/master/transformers/report.tex)
