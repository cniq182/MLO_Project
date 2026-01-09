## Machine Learning Operations Project

Sarah Tauro - s243927
Jorge Ernesto Figueroa Vallecillo - s250273
Álvaro Quintana López - s250202
Víctor Cabré i Guerrero - s254631
Cesia Niquen Tapia - s250829


# Final Summary for hand-in: 

# Overall goal of the project

The goal of this project is to develop a machine translation system from English to Spanish. This language pair was selected because four out of five group members are Spanish speakers, which facilitates qualitative evaluation and error analysis of the translations. The project focuses on exploring how transformer-based architectures can be adapted for machine translation.

# Frameworks

The framework used in this project is the Transformer architecture. We will incorporate this framework by leveraging pretrained transformer-based language models and fine-tuning them on English–Spanish data. The project pipeline includes data loading, preprocessing, tokenization, model fine-tuning, and performance evaluation using standard translation metrics.

# Data

For training and evaluation, we will use data from the OPUS collection, a widely used open-source resource for machine translation research. Specifically, we will work with the OpenSubtitles corpus, which provides large-scale English–Spanish parallel sentence pairs derived from movie and television subtitles. This dataset is well suited for conversational and informal language translation tasks.

# Models 

In terms of models, we plan to experiment with pretrained transformer-based NLP models that were not explicitly trained for translation. In particular, we will consider BERT-based models and T5. BERT is an encoder-only model trained on large-scale text for language understanding tasks, while T5 follows a sequence-to-sequence (encoder–decoder) architecture that aligns with translation objectives. By fine-tuning and comparing these models, we aim to evaluate how different transformer architectures perform when adapted to machine translation.


