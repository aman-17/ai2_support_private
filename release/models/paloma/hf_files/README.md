---
extra_gated_prompt: "Access to this model is automatically granted upon accepting the [**AI2 ImpACT License – Low Risk Artifacts (“LR Agreement”)**](https://allenai.org/licenses/impact-lr) and completing all fields below. This model is licensed under the LR Agreement."
extra_gated_fields:
 Your full name: text
 Organization or entity you are affiliated with: text
 State or country you are located in: text
 Contact email: text
 Please describe your intended use of the low risk artifact(s): text
 I AGREE to the terms and conditions of the LR Agreement above: checkbox
 I AGREE to AI2’s use of my information for legal notices and administrative matters: checkbox
 I CERTIFY that the information I have provided is true and accurate: checkbox

# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards

---

# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->

The [Paloma](https://paloma.allen.ai/) 1B baselines are a collection of language models pretrained on popular corpora while controlling all other experimental variables. These models are developed to facilitate scientific comparisons of language model fit using the Paloma benchmark of 585 textual domains. This collection of models includes 6 baseline 1B parameter models each trained on ~150B tokens from one the following corpora: [Dolma](https://github.com/allenai/dolma), [The Pile](https://api.semanticscholar.org/CorpusID:230435736), [RedPajama](https://github.com/togethercomputer/RedPajama-Data), [Falcon-RefinedWeb](https://api.semanticscholar.org/CorpusID:259063761), [C4](https://aclanthology.org/2021.emnlp-main.98), and [MC4-en](https://api.semanticscholar.org/CorpusID:258187051).

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** Ian Magnusson, Akshita Bhagia, Valentin Hofmann, Luca Soldaini, Ananya Harsh Jha, Oyvind Tafjord, Dustin Schwenk, Evan Pete Walsh, Yanai Elazar, Kyle Lo, Dirk Groeneveld, Iz Beltagy, Hannaneh Hajishirzi, Noah A. Smith, Kyle Richardson, and Jesse Dodge
- **Model type:** Decoder-only transformer language model
- **Language(s) (NLP):** English
- **License:** [**AI2 ImpACT License – Low Risk Artifacts (“LR Agreement”)**](https://allenai.org/licenses/impact-lr)

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

This model is primarily intended as research artifact that is a baseline for the language modeling benchmark [Paloma](https://paloma.allen.ai/).

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

The restrictions to use of this model are described in the model license: [**AI2 ImpACT License – Low Risk Artifacts (“LR Agreement”)**](https://allenai.org/licenses/impact-lr)

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

This model is purely trained as an autoregressive language model. It has not been adapted in any way to prevent bias. It is a model of the language distribution that it is trained on.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

This research artifact is a baseline for a language modeling benchmark. Best uses of this model will take advantage of the experimental controls applied to this model and the other Paloma baselines. These enable comparisons of models that vary only in the pretraining corpus used to train them.

## How to Get Started with the Model

Install the code needed to run inference with the model
```
pip install ai2-olmo
```

Download and instantiate the model
```
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("allenai/<model name here>", trust_remote_code=True)
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

Each of the Paloma baseline models are trained on one of the following datasets:
[Dolma](https://github.com/allenai/dolma), [The Pile](https://api.semanticscholar.org/CorpusID:230435736), [RedPajama](https://github.com/togethercomputer/RedPajama-Data), [Falcon-RefinedWeb](https://api.semanticscholar.org/CorpusID:259063761), [C4](https://aclanthology.org/2021.emnlp-main.98), and [MC4-en](https://api.semanticscholar.org/CorpusID:258187051).

### Training Procedure 

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing

We remove any document in the pretraining data that is contaminated with respect to the [Paloma](https://paloma.allen.ai/) evaluation data. We match overlaps of evaluation and train text at the paragraph level, i.e., newline separated spans of text. To avoid coincidental collisions in the space of small strings, we ignore matches in paragraphs smaller than 13 unicode segmented tokens. Similarly, we ignore paragraphs composed of only punctuation, spaces, and emoji, as, unlike words, these can be arbitrarily repeated when used as formatting, leading to high frequency n-grams greater than our 13-gram threshold. Lastly, as code data consists almost entirely of short and often repeated lines, we forgo any decontamination against the code evaluations in Paloma.


#### Training Hyperparameters

The Paloma baseline 1B parameter models that we train employ the following architecture: 2048 maximum sequence length, 2048 model dimension, 16 layers, 16 attention heads, RoPE embedding, SwiGLU activation, mixed precision, non-parametric layer normalization, and sequential model blocks for attention and feed-forward networks. 
We use EleutherAI's GPT NeoX tokenizer but add 3 additional special tokens that are used to mask PII in Dolma.
We train to 35k steps (∼150B tokens) with the following LionW optimizer configurations: 2.0e-4 peak learning rate, warm-up of 2000 steps, cosine decay to 70k steps (∼300B tokens), 0.1 weight decay, and betas of 0.9 and 0.95. Note that our batch size varies slightly to accommodate two groups of baselines that were run on different hardware. The Dolma and Falcon-RefinedWeb baselines were run with a batch size of 2112 training instances per step on 24 A100s. The RedPajama, The Pile, C4, and mC4-EN baselines were run with a batch size of 2048 on 64 AMD Instinct MI250X GPUs. In each case we save model checkpoints every 5k steps (∼20B tokens).

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

The [Paloma](https://paloma.allen.ai/) benchmark is used to evaluate these baseline models.

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

Paloma evaluates on 585 domains. These are a collection of the most fine-grained domains readily available in current metadata.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

Paloma measures langauge modeling fit using Perplexity. It is a benchmark of language modeling, so examination of downstream uses is out of scope.

### Results

To demonstrate possible uses of results from the Paloma benchmark, we conduct a series of case studies. We show that performance improves in almost all domains as models are scaled, but domains improve unequally. Further, across domains, perplexity is driven by strings in the vocabulary, i.e., types, that occur in most domains, but other types even get worse as models scale. Finally, our experiments isolate change in pretraining corpora and find that pretraining without heterogeneous data sources beyond Common Crawl leads to perplexities that do not improve consistently with tokens seen.


## Environmental Impact

The Dolma and Falcon-RefinedWeb baselines were run with on 24 A100s for 9 days per model.
The RedPajama, The Pile, C4, and mC4-EN baselines were run on 64 AMD Instinct MI250X GPUs for 2 days per model.


## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```
@article{paloma,
  title={{Paloma}: A Benchmark for Evaluating Language Model Fit},
  author={Magnusson, Ian and Bhagia, Akshita and Hofmann, Valentin and Soldaini, Luca and Harsh Jha, Ananya and Tafjord, Oyvind and Schwenk,Dustin and Walsh, Evan Pete and Elazar, Yanai and Lo, Kyle and Groenveld,Dirk and Beltagy,Iz and  Hajishirz,Hanneneh and Smith, Noah A. and Richardson,Kyle and Dodge,Jesse},
  journal={technical report},
  year={2023},
  url={https://paloma.allen.ai/}
} 
```


## Model Card Contact

{ianm,jessed}@allenai.org
