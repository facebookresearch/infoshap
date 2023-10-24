# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import shap
import transformers
import nlp
import torch
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import webbrowser

# load a BERT sentiment analysis model
tokenizer = transformers.DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = transformers.DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
).cuda()

def entropy(x):
    _x = x
    logp = np.log(_x)
    plogp = np.multiply(_x, logp)
    out = np.sum(plogp, axis=1)
    return -out


def f(x):
    tv = torch.tensor([tokenizer.encode(v, pad_to_max_length=True, max_length=500) for v in x]).cuda()
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = entropy(scores)
    #val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    return val

imdb_train = nlp.load_dataset("imdb")["train"]

background = 1000
test_reviews = []
for i in range(background):
    if len(imdb_train[i]['text']) < 300:
        test_reviews.append(i)

subset = imdb_train.select(test_reviews)

#build an explainer using a token masker
explainer = shap.Explainer(f, tokenizer)

# explain the model's predictions on IMDB reviews
shap_values = explainer(subset)

for i in range(len(shap_values)):
    uncertainty = shap_values[i].base_values + np.sum(shap_values[i].values)
    if uncertainty > 0.5:
        file = open(str(i) + '.html','w')
        file.write(shap.plots.text(shap_values[i], display=False))
        file.close
