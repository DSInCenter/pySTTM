# Topic Modeling Tool for Persian Short Texts

The tool for topic modeling provided by the **[Data Science Innovation Center](http://dslab.aut.ac.ir/fa/)** extracts topics from digitized **Persian texts** and compares their performance in short texts using a variety of topic modeling techniques.

Visit the **[website](http://dslab.aut.ac.ir/fa/products/%d9%be%d8%b1%d8%af%d8%a7%d8%b2%d8%b4-%d9%85%d8%aa%d9%86-%d9%88-%d8%b2%d8%a8%d8%a7%d9%86-%d8%b7%d8%a8%db%8c%d8%b9%db%8c/%d8%a7%d8%a8%d8%b2%d8%a7%d8%b1-%d8%af%d8%b3%d8%aa%d9%87-%d8%a8%d9%86%d8%af%db%8c-%d9%85%d9%88%d8%b6%d9%88%d8%b9%db%8c/)** to view the description in Persian.

## Installation
We recommend **Python 3.6** or higher, **[gensim 4.2](https://radimrehurek.com/gensim/)** or higher.

**Install from sources**

You can also clone the latest version from the repository and install it directly from the source code:

```
git clone https://github.com/DSInCenter/topicmodel.git
cd topicmodel
pip install -r requirements.txt
```

## Getting Started
These examples demonstrate how to clone and execute a model on Google Colab:
- [Run NMF model on Google colab](https://colab.research.google.com/drive/1l7Fs6yYrbIy9fXyTBflMXGaVQjh10RPn?usp=sharing)
- [Run LDA model on Google colab](https://colab.research.google.com/drive/1yhNeh6J177fSQxEZE7OTLJMWtvff7LDA?usp=sharing) 

**LDA demonstration**:

First, import Dataset Class from Dataset.py and import LDA model from LDA.py:
````python
from tools.Dataset import Dataset
from LDA import LDA
````

Create Objects from Dataset and LDA Classes and Traing The Model:
````python
lda = LDA(num_topics=11, iterations=5)
dataset = Dataset('Dataset', 'utf-8')
lda_result = lda.train_model(dataset, hyperparams=None, top_words=10)
print(lda_result)
````

## Citing & Authors
If you find this repository helpful, feel free to cite this work []():

```bibtex 
@inproceedings{
}
```

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

