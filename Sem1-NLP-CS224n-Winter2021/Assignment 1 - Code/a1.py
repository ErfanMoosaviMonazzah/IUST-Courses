# %% [markdown]
# # CS224N Assignment 1: Exploring Word Vectors (25 Points)
# ### <font color='blue'> Due 3:15pm, Tue Jan 11 </font>
# 
# Welcome to CS224N! 
# 
# Before you start, make sure you read the README.txt in the same directory as this notebook for important setup information. A lot of code is provided in this notebook, and we highly encourage you to read and understand it as part of the learning :)
# 
# If you aren't super familiar with Python, Numpy, or Matplotlib, we recommend you check out the review session on Friday. The session will be recorded and the material will be made available on our [website](http://web.stanford.edu/class/cs224n/index.html#schedule). The CS231N Python/Numpy [tutorial](https://cs231n.github.io/python-numpy-tutorial/) is also a great resource.
# 
# 
# **Assignment Notes:** Please make sure to save the notebook as you go along. Submission Instructions are located at the bottom of the notebook.

# %%
# All Import Statements Defined Here
# Note: Do not add to this list.
# ----------------

import sys
assert sys.version_info[0]==3
assert sys.version_info[1] >= 5

from platform import python_version
assert int(python_version().split(".")[1]) >= 5, "Please upgrade your Python version following the instructions in \
    the README.txt file found in the same directory as this notebook. Your Python version is " + python_version()

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pprint
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
import nltk
nltk.download('reuters') #to specify download location, optionally add the argument: download_dir='/specify/desired/path/'
from nltk.corpus import reuters
import numpy as np
import random
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)
# ----------------

# %% [markdown]
# ## Word Vectors
# 
# Word Vectors are often used as a fundamental component for downstream NLP tasks, e.g. question answering, text generation, translation, etc., so it is important to build some intuitions as to their strengths and weaknesses. Here, you will explore two types of word vectors: those derived from *co-occurrence matrices*, and those derived via *GloVe*. 
# 
# **Note on Terminology:** The terms "word vectors" and "word embeddings" are often used interchangeably. The term "embedding" refers to the fact that we are encoding aspects of a word's meaning in a lower dimensional space. As [Wikipedia](https://en.wikipedia.org/wiki/Word_embedding) states, "*conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension*".

# %% [markdown]
# ## Part 1: Count-Based Word Vectors (10 points)
# 
# Most word vector models start from the following idea:
# 
# *You shall know a word by the company it keeps ([Firth, J. R. 1957:11](https://en.wikipedia.org/wiki/John_Rupert_Firth))*
# 
# Many word vector implementations are driven by the idea that similar words, i.e., (near) synonyms, will be used in similar contexts. As a result, similar words will often be spoken or written along with a shared subset of words, i.e., contexts. By examining these contexts, we can try to develop embeddings for our words. With this intuition in mind, many "old school" approaches to constructing word vectors relied on word counts. Here we elaborate upon one of those strategies, *co-occurrence matrices* (for more information, see [here](https://web.stanford.edu/~jurafsky/slp3/6.pdf) or [here](https://medium.com/data-science-group-iitr/word-embedding-2d05d270b285)).

# %% [markdown]
# ### Co-Occurrence
# 
# A co-occurrence matrix counts how often things co-occur in some environment. Given some word $w_i$ occurring in the document, we consider the *context window* surrounding $w_i$. Supposing our fixed window size is $n$, then this is the $n$ preceding and $n$ subsequent words in that document, i.e. words $w_{i-n} \dots w_{i-1}$ and $w_{i+1} \dots w_{i+n}$. We build a *co-occurrence matrix* $M$, which is a symmetric word-by-word matrix in which $M_{ij}$ is the number of times $w_j$ appears inside $w_i$'s window among all documents.
# 
# **Example: Co-Occurrence with Fixed Window of n=1**:
# 
# Document 1: "all that glitters is not gold"
# 
# Document 2: "all is well that ends well"
# 
# 
# |     *    | `<START>` | all | that | glitters | is   | not  | gold  | well | ends | `<END>` |
# |----------|-------|-----|------|----------|------|------|-------|------|------|-----|
# | `<START>`    | 0     | 2   | 0    | 0        | 0    | 0    | 0     | 0    | 0    | 0   |
# | all      | 2     | 0   | 1    | 0        | 1    | 0    | 0     | 0    | 0    | 0   |
# | that     | 0     | 1   | 0    | 1        | 0    | 0    | 0     | 1    | 1    | 0   |
# | glitters | 0     | 0   | 1    | 0        | 1    | 0    | 0     | 0    | 0    | 0   |
# | is       | 0     | 1   | 0    | 1        | 0    | 1    | 0     | 1    | 0    | 0   |
# | not      | 0     | 0   | 0    | 0        | 1    | 0    | 1     | 0    | 0    | 0   |
# | gold     | 0     | 0   | 0    | 0        | 0    | 1    | 0     | 0    | 0    | 1   |
# | well     | 0     | 0   | 1    | 0        | 1    | 0    | 0     | 0    | 1    | 1   |
# | ends     | 0     | 0   | 1    | 0        | 0    | 0    | 0     | 1    | 0    | 0   |
# | `<END>`      | 0     | 0   | 0    | 0        | 0    | 0    | 1     | 1    | 0    | 0   |
# 
# **Note:** In NLP, we often add `<START>` and `<END>` tokens to represent the beginning and end of sentences, paragraphs or documents. In this case we imagine `<START>` and `<END>` tokens encapsulating each document, e.g., "`<START>` All that glitters is not gold `<END>`", and include these tokens in our co-occurrence counts.
# 
# The rows (or columns) of this matrix provide one type of word vectors (those based on word-word co-occurrence), but the vectors will be large in general (linear in the number of distinct words in a corpus). Thus, our next step is to run *dimensionality reduction*. In particular, we will run *SVD (Singular Value Decomposition)*, which is a kind of generalized *PCA (Principal Components Analysis)* to select the top $k$ principal components. Here's a visualization of dimensionality reduction with SVD. In this picture our co-occurrence matrix is $A$ with $n$ rows corresponding to $n$ words. We obtain a full matrix decomposition, with the singular values ordered in the diagonal $S$ matrix, and our new, shorter length-$k$ word vectors in $U_k$.
# 
# ![Picture of an SVD](imgs/svd.png "SVD")
# 
# This reduced-dimensionality co-occurrence representation preserves semantic relationships between words, e.g. *doctor* and *hospital* will be closer than *doctor* and *dog*. 
# 
# **Notes:** If you can barely remember what an eigenvalue is, here's [a slow, friendly introduction to SVD](https://davetang.org/file/Singular_Value_Decomposition_Tutorial.pdf). If you want to learn more thoroughly about PCA or SVD, feel free to check out lectures [7](https://web.stanford.edu/class/cs168/l/l7.pdf), [8](http://theory.stanford.edu/~tim/s15/l/l8.pdf), and [9](https://web.stanford.edu/class/cs168/l/l9.pdf) of CS168. These course notes provide a great high-level treatment of these general purpose algorithms. Though, for the purpose of this class, you only need to know how to extract the k-dimensional embeddings by utilizing pre-programmed implementations of these algorithms from the numpy, scipy, or sklearn python packages. In practice, it is challenging to apply full SVD to large corpora because of the memory needed to perform PCA or SVD. However, if you only want the top $k$ vector components for relatively small $k$ — known as [Truncated SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition#Truncated_SVD) — then there are reasonably scalable techniques to compute those iteratively.

# %% [markdown]
# ### Plotting Co-Occurrence Word Embeddings
# 
# Here, we will be using the Reuters (business and financial news) corpus. If you haven't run the import cell at the top of this page, please run it now (click it and press SHIFT-RETURN). The corpus consists of 10,788 news documents totaling 1.3 million words. These documents span 90 categories and are split into train and test. For more details, please see https://www.nltk.org/book/ch02.html. We provide a `read_corpus` function below that pulls out only articles from the "grain" (i.e. news articles about corn, wheat, etc.) category. The function also adds `<START>` and `<END>` tokens to each of the documents, and lowercases words. You do **not** have to perform any other kind of pre-processing.

# %%
def read_corpus(category="grain"):
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]


# %% [markdown]
# Let's have a look what these documents are like….

# %%
reuters_corpus = read_corpus()
pprint.pprint(reuters_corpus[:3], compact=True, width=100)

# %% [markdown]
# ### Question 1.1: Implement `distinct_words` [code] (2 points)
# 
# Write a method to work out the distinct words (word types) that occur in the corpus. You can do this with `for` loops, but it's more efficient to do it with Python list comprehensions. In particular, [this](https://coderwall.com/p/rcmaea/flatten-a-list-of-lists-in-one-line-in-python) may be useful to flatten a list of lists. If you're not familiar with Python list comprehensions in general, here's [more information](https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Comprehensions.html).
# 
# Your returned `corpus_words` should be sorted. You can use python's `sorted` function for this.
# 
# You may find it useful to use [Python sets](https://www.w3schools.com/python/python_sets.asp) to remove duplicate words.

# %%
def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): sorted list of distinct words across the corpus
            n_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    n_corpus_words = -1
    
    # ------------------
    # Write your implementation here.
    corpus_words = sorted(list({word for wordlist in corpus for word in wordlist}))
    n_corpus_words = len(corpus_words)
    # ------------------

    return corpus_words, n_corpus_words

# %%
# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness.
# ---------------------

# Define toy corpus
test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "), "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
test_corpus_words, num_corpus_words = distinct_words(test_corpus)

# Correct answers
ans_test_corpus_words = sorted([START_TOKEN, "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", END_TOKEN])
ans_num_corpus_words = len(ans_test_corpus_words)

# Test correct number of words
assert(num_corpus_words == ans_num_corpus_words), "Incorrect number of distinct words. Correct: {}. Yours: {}".format(ans_num_corpus_words, num_corpus_words)

# Test correct words
assert (test_corpus_words == ans_test_corpus_words), "Incorrect corpus_words.\nCorrect: {}\nYours:   {}".format(str(ans_test_corpus_words), str(test_corpus_words))

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)

# %% [markdown]
# ### Question 1.2: Implement `compute_co_occurrence_matrix` [code] (3 points)
# 
# Write a method that constructs a co-occurrence matrix for a certain window-size $n$ (with a default of 4), considering words $n$ before and $n$ after the word in the center of the window. Here, we start to use `numpy (np)` to represent vectors, matrices, and tensors. If you're not familiar with NumPy, there's a NumPy tutorial in the second half of this cs231n [Python NumPy tutorial](http://cs231n.github.io/python-numpy-tutorial/).
# 

# %%
def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).

        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.

              For example, if we take the document "<START> All that glitters is not gold <END>" with window size of 4,
              "All" will co-occur with "<START>", "that", "glitters", "is", and "not".

        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (a symmetric numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)):
                Co-occurence matrix of word counts.
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, n_words = distinct_words(corpus)
    M = None
    word2ind = {}

    # ------------------
    # Write your implementation here.
    word2ind = {k:v for k, v in zip(words, range(n_words))}
    M = np.zeros((n_words, n_words))
    
    for doc in corpus:
        for i, word in enumerate(doc):
            contexts = doc[i+1:min(len(doc), i + window_size + 1)]
            indx_contexts = [word2ind[w] for w in contexts]
            M[word2ind[word], indx_contexts] += 1
            M[indx_contexts, word2ind[word]] += 1
    
    # ------------------

    return M, word2ind

# %%
# ---------------------
# Run this sanity check
# Note that this is not an exhaustive check for correctness.
# ---------------------

# Define toy corpus and get student's co-occurrence matrix
test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "), "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
M_test, word2ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)

# Correct M and word2ind
M_test_ans = np.array( 
    [[0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,],
     [0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,],
     [0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,],
     [0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,],
     [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,],
     [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,],
     [1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,],
     [0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,],
     [0., 0., 1., 0., 1., 1., 0., 0., 0., 1.,],
     [1., 0., 0., 1., 1., 0., 0., 0., 1., 0.,]]
)
ans_test_corpus_words = sorted([START_TOKEN, "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", END_TOKEN])
word2ind_ans = dict(zip(ans_test_corpus_words, range(len(ans_test_corpus_words))))

# Test correct word2ind
assert (word2ind_ans == word2ind_test), "Your word2ind is incorrect:\nCorrect: {}\nYours: {}".format(word2ind_ans, word2ind_test)

# Test correct M shape
assert (M_test.shape == M_test_ans.shape), "M matrix has incorrect shape.\nCorrect: {}\nYours: {}".format(M_test.shape, M_test_ans.shape)

# Test correct M values
for w1 in word2ind_ans.keys():
    idx1 = word2ind_ans[w1]
    for w2 in word2ind_ans.keys():
        idx2 = word2ind_ans[w2]
        student = M_test[idx1, idx2]
        correct = M_test_ans[idx1, idx2]
        if student != correct:
            print("Correct M:")
            print(M_test_ans)
            print("Your M: ")
            print(M_test)
            raise AssertionError("Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(idx1, idx2, w1, w2, student, correct))

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)

# %% [markdown]
# ### Question 1.3: Implement `reduce_to_k_dim` [code] (1 point)
# 
# Construct a method that performs dimensionality reduction on the matrix to produce k-dimensional embeddings. Use SVD to take the top k components and produce a new matrix of k-dimensional embeddings. 
# 
# **Note:** All of numpy, scipy, and scikit-learn (`sklearn`) provide *some* implementation of SVD, but only scipy and sklearn provide an implementation of Truncated SVD, and only sklearn provides an efficient randomized algorithm for calculating large-scale Truncated SVD. So please use [sklearn.decomposition.TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).

# %%
def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
        Params:
            M (numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """    
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    
    # ------------------
    # Write your implementation here.
    tsvd = TruncatedSVD(n_components=k, n_iter=n_iters)
    M_reduced = tsvd.fit_transform(M)
    

    # ------------------

    print("Done.")
    return M_reduced

# %%
# ---------------------
# Run this sanity check
# Note that this is not an exhaustive check for correctness 
# In fact we only check that your M_reduced has the right dimensions.
# ---------------------

# Define toy corpus and run student code
test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "), "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
M_test, word2ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)
M_test_reduced = reduce_to_k_dim(M_test, k=2)

# Test proper dimensions
assert (M_test_reduced.shape[0] == 10), "M_reduced has {} rows; should have {}".format(M_test_reduced.shape[0], 10)
assert (M_test_reduced.shape[1] == 2), "M_reduced has {} columns; should have {}".format(M_test_reduced.shape[1], 2)

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)

# %% [markdown]
# ### Question 1.4: Implement `plot_embeddings` [code] (1 point)
# 
# Here you will write a function to plot a set of 2D vectors in 2D space. For graphs, we will use Matplotlib (`plt`).
# 
# For this example, you may find it useful to adapt [this code](http://web.archive.org/web/20190924160434/https://www.pythonmembers.club/2018/05/08/matplotlib-scatter-plot-annotate-set-text-at-label-each-point/). In the future, a good way to make a plot is to look at [the Matplotlib gallery](https://matplotlib.org/gallery/index.html), find a plot that looks somewhat like what you want, and adapt the code they give.

# %%
def plot_embeddings(M_reduced, word2ind, words):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2ind.
        Include a label next to each point.
        
        Params:
            M_reduced (numpy matrix of shape (number of unique words in the corpus , 2)): matrix of 2-dimensioal word embeddings
            word2ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """

    # ------------------
    # Write your implementation here.
    for word in words:
        i = word2ind[word]
        x, y = M_reduced[i, :]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x, y, word, fontsize=9)
    plt.show()
    # ------------------

# %%
# ---------------------
# Run this sanity check
# Note that this is not an exhaustive check for correctness.
# The plot produced should look like the "test solution plot" depicted below. 
# ---------------------

print ("-" * 80)
print ("Outputted Plot:")

M_reduced_plot_test = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]])
word2ind_plot_test = {'test1': 0, 'test2': 1, 'test3': 2, 'test4': 3, 'test5': 4}
words = ['test1', 'test2', 'test3', 'test4', 'test5']
plot_embeddings(M_reduced_plot_test, word2ind_plot_test, words)

print ("-" * 80)

# %% [markdown]
# <font color=red>**Test Plot Solution**</font>
# <br>
# <img src="imgs/test_plot.png" width=40% style="float: left;"> </img>
# 

# %% [markdown]
# ### Question 1.5: Co-Occurrence Plot Analysis [written] (3 points)
# 
# Now we will put together all the parts you have written! We will compute the co-occurrence matrix with fixed window of 4 (the default window size), over the Reuters "grain" corpus. Then we will use TruncatedSVD to compute 2-dimensional embeddings of each word. TruncatedSVD returns U\*S, so we need to normalize the returned vectors, so that all the vectors will appear around the unit circle (therefore closeness is directional closeness). **Note**: The line of code below that does the normalizing uses the NumPy concept of *broadcasting*. If you don't know about broadcasting, check out
# [Computation on Arrays: Broadcasting by Jake VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html).
# 
# Run the below cell to produce the plot. It'll probably take a few seconds to run. What clusters together in 2-dimensional embedding space? What doesn't cluster together that you might think should have?

# %%
# -----------------------------
# Run This Cell to Produce Your Plot
# ------------------------------
reuters_corpus = read_corpus()
M_co_occurrence, word2ind_co_occurrence = compute_co_occurrence_matrix(reuters_corpus)
M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)

# Rescale (normalize) the rows to make them each of unit-length
M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis] # broadcasting

words = ['tonnes', 'grain', 'wheat',  'agriculture', 'corn', 'maize', 'export', 'department', 'barley', 'grains', 'soybeans', 'sorghum']

plot_embeddings(M_normalized, word2ind_co_occurrence, words)

# %% [markdown]
# #### <font color="red">Write your answer here.</font>
# ## [Erfan:]
# Cluser 1:
# - corn
# - sorghum
# - grain
# 
# Cluster 2:
# - wheat
# - export
# 
# Cluster 3: (or u can say cluster 2 and 3 are in one big cluster)
# - grains
# - tonnes
# 
# Cluster 4:
# - barley
# - maize
# - soybeans
# 
# Cluster 5??: (I think they are close enough to be considered as a cluster)
# - agriculture 
# - department
# 
# Since maize and corn are the same thing, they can be used in the same context. But here in this plot, they are in different clusters.
# I expected barley to be near wheat, because they usually come in same context.

# %% [markdown]
# ## Part 2: Prediction-Based Word Vectors (15 points)
# 
# As discussed in class, more recently prediction-based word vectors have demonstrated better performance, such as word2vec and GloVe (which also utilizes the benefit of counts). Here, we shall explore the embeddings produced by GloVe. Please revisit the class notes and lecture slides for more details on the word2vec and GloVe algorithms. If you're feeling adventurous, challenge yourself and try reading [GloVe's original paper](https://nlp.stanford.edu/pubs/glove.pdf).
# 
# Then run the following cells to load the GloVe vectors into memory. **Note**: If this is your first time to run these cells, i.e. download the embedding model, it will take a couple minutes to run. If you've run these cells before, rerunning them will load the model without redownloading it, which will take about 1 to 2 minutes.

# %%
def load_embedding_model():
    """ Load GloVe Vectors
        Return:
            wv_from_bin: All 400000 embeddings, each lengh 200
    """
    import gensim.downloader as api
    wv_from_bin = api.load("glove-wiki-gigaword-200")
    print("Loaded vocab size %i" % len(list(wv_from_bin.index_to_key)))
    return wv_from_bin

# %%
# -----------------------------------
# Run Cell to Load Word Vectors
# Note: This will take a couple minutes (previously took 1m 6.7s)
# -----------------------------------
wv_from_bin = load_embedding_model()

# %% [markdown]
# #### Note: If you are receiving a "reset by peer" error, rerun the cell to restart the download. 

# %% [markdown]
# ### Reducing dimensionality of Word Embeddings
# Let's directly compare the GloVe embeddings to those of the co-occurrence matrix. In order to avoid running out of memory, we will work with a sample of 10000 GloVe vectors instead.
# Run the following cells to:
# 
# 1. Put 10000 Glove vectors into a matrix M
# 2. Run `reduce_to_k_dim` (your Truncated SVD function) to reduce the vectors from 200-dimensional to 2-dimensional.

# %%
def get_matrix_of_vectors(wv_from_bin, required_words=['tonnes', 'grain', 'wheat',  'agriculture', 'corn', 'maize', 'export', 'department', 'barley', 'grains', 'soybeans', 'sorghum']):
    """ Put the GloVe vectors into a matrix M.
        Param:
            wv_from_bin: KeyedVectors object; the 400000 GloVe vectors loaded from file
        Return:
            M: numpy matrix shape (num words, 200) containing the vectors
            word2ind: dictionary mapping each word to its row number in M
    """
    import random
    words = list(wv_from_bin.index_to_key)
    print("Shuffling words ...")
    random.seed(225)
    random.shuffle(words)
    words = words[:10000]
    print("Putting %i words into word2ind and matrix M..." % len(words))
    word2ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            M.append(wv_from_bin.get_vector(w))
            word2ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        if w in words:
            continue
        try:
            M.append(wv_from_bin.get_vector(w))
            word2ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    M = np.stack(M)
    print("Done.")
    return M, word2ind

# %%
# -----------------------------------------------------------------
# Run Cell to Reduce 200-Dimensional Word Embeddings to k Dimensions
# Note: This should be quick to run
# -----------------------------------------------------------------
M, word2ind = get_matrix_of_vectors(wv_from_bin)
M_reduced = reduce_to_k_dim(M, k=2)

# Rescale (normalize) the rows to make them each of unit-length
M_lengths = np.linalg.norm(M_reduced, axis=1)
M_reduced_normalized = M_reduced / M_lengths[:, np.newaxis] # broadcasting

# %% [markdown]
# **Note: If you are receiving out of memory issues on your local machine, try closing other applications to free more memory on your device. You may want to try restarting your machine so that you can free up extra memory. Then immediately run the jupyter notebook and see if you can load the word vectors properly. If you still have problems with loading the embeddings onto your local machine after this, please go to office hours or contact course staff.**

# %% [markdown]
# ### Question 2.1: GloVe Plot Analysis [written] (3 points)
# 
# Run the cell below to plot the 2D GloVe embeddings for `['tonnes', 'grain', 'wheat',  'agriculture', 'corn', 'maize', 'export', 'department', 'barley', 'grains', 'soybeans', 'sorghum']`.
# 
# What clusters together in 2-dimensional embedding space? What doesn't cluster together that you think should have? How is the plot different from the one generated earlier from the co-occurrence matrix? What is a possible cause for the difference?

# %%
words = ['tonnes', 'grain', 'wheat',  'agriculture', 'corn', 'maize', 'export', 'department', 'barley', 'grains', 'soybeans', 'sorghum']
plot_embeddings(M_reduced_normalized, word2ind, words)

# %% [markdown]
# #### <font color="red">Write your answer here.</font>
# ## [Erfan]: (Since this time clusters are smooshed together, I used the dynamic plot generated by python interpreter to be able to zoom in)
# Cluster 1
# - wheat
# - corn
# - grain
# - export
# 
# Cluster 2
# - grains
# - tonnes
# - maize
# - soybeans
# - barley
# 
# Cluster 4
# - sorghum
# 
# Cluster 5?? (I think they are close enough to be considered as a cluster)
# - agriculture
# - department
# 
# Since maize and corn are the same thing, they can be used in the same context. But here in this plot, they are in different clusters.
# I expected barley to be near wheat, because they usually come in same context.
# 
# How's differ from previous one:
# In this embedding, corn and grain are clustred with wheat and export.
# maize, soybeans and barley are still in the same cluster, but now grains and tonnes are in this cluster too.
# sorghum and corn was in the same cluster in previous embedding but know sorghum is in its own cluster.
# 
# Why is that differ?
# Glove embedding is done on bigger corpus, and this may cause the difference between the context of words in each of these embeddings. Basic principle for these embeddings is that they use the neighborhood of a word to determine its context. So if a word has different neighbors in different corpus, the embeddings are different too.

# %% [markdown]
# ### Cosine Similarity
# Now that we have word vectors, we need a way to quantify the similarity between individual words, according to these vectors. One such metric is cosine-similarity. We will be using this to find words that are "close" and "far" from one another.
# 
# We can think of n-dimensional vectors as points in n-dimensional space. If we take this perspective [L1](http://mathworld.wolfram.com/L1-Norm.html) and [L2](http://mathworld.wolfram.com/L2-Norm.html) Distances help quantify the amount of space "we must travel" to get between these two points. Another approach is to examine the angle between two vectors. From trigonometry we know that:
# 
# <img src="imgs/inner_product.png" width=20% style="float: center;"></img>
# 
# Instead of computing the actual angle, we can leave the similarity in terms of $similarity = cos(\Theta)$. Formally the [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity) $s$ between two vectors $p$ and $q$ is defined as:
# 
# $$s = \frac{p \cdot q}{||p|| ||q||}, \textrm{ where } s \in [-1, 1] $$ 

# %% [markdown]
# ### Question 2.2: Words with Multiple Meanings (1.5 points) [code + written] 
# Polysemes and homonyms are words that have more than one meaning (see this [wiki page](https://en.wikipedia.org/wiki/Polysemy) to learn more about the difference between polysemes and homonyms ). Find a word with *at least two different meanings* such that the top-10 most similar words (according to cosine similarity) contain related words from *both* meanings. For example, "leaves" has both "go_away" and "a_structure_of_a_plant" meaning in the top 10, and "scoop" has both "handed_waffle_cone" and "lowdown". You will probably need to try several polysemous or homonymic words before you find one. 
# 
# Please state the word you discover and the multiple meanings that occur in the top 10. Why do you think many of the polysemous or homonymic words you tried didn't work (i.e. the top-10 most similar words only contain **one** of the meanings of the words)?
# 
# **Note**: You should use the `wv_from_bin.most_similar(word)` function to get the top 10 similar words. This function ranks all other words in the vocabulary with respect to their cosine similarity to the given word. For further assistance, please check the __[GenSim documentation](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.FastTextKeyedVectors.most_similar)__.

# %%
# ------------------
# Write your implementation here.
# wv_from_bin.most_similar('exhaust') # usage, tiring, car smoke pipe
# wv_from_bin.most_similar('spring') # فنر، بهار، چشمه
wv_from_bin.most_similar('fall') # season, falling ### good
wv_from_bin.most_similar('scope') # zoom of gun, borders and limitations ### good
# wv_from_bin.most_similar('galaxy') # brand, cosmos
wv_from_bin.most_similar('pretty') # appearance, quite 
# ------------------

# %% [markdown]
# #### <font color="red">Write your answer here.</font>
# ## [Erfan:]
# 
# I think the reason behind this is that on the corpus which was used to train this embedding, one of the meanings of the words was more dominant than the other, so since these embeddings created based on the context of the word (which means the negihboring words), this dominance cause such thing.

# %% [markdown]
# ### Question 2.3: Synonyms & Antonyms (2 points) [code + written] 
# 
# When considering Cosine Similarity, it's often more convenient to think of Cosine Distance, which is simply 1 - Cosine Similarity.
# 
# Find three words $(w_1,w_2,w_3)$ where $w_1$ and $w_2$ are synonyms and $w_1$ and $w_3$ are antonyms, but Cosine Distance $(w_1,w_3) <$ Cosine Distance $(w_1,w_2)$. 
# 
# As an example, $w_1$="happy" is closer to $w_3$="sad" than to $w_2$="cheerful". Please find a different example that satisfies the above. Once you have found your example, please give a possible explanation for why this counter-intuitive result may have happened.
# 
# You should use the the `wv_from_bin.distance(w1, w2)` function here in order to compute the cosine distance between two words. Please see the __[GenSim documentation](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.FastTextKeyedVectors.distance)__ for further assistance.

# %%
# ------------------
# Write your implementation here.

# (cold, cool, warm) ### Good
wv_from_bin.distance('cold', 'warm')
wv_from_bin.distance('cold', 'cool')

# ------------------

# %% [markdown]
# #### <font color="red">Write your answer here.</font>
# ## [Erfan:]
# 
# When we talk aboud embeddings, we are considering context, sometimes it is normal for two acronyms to be in the same context, for example, sad and happy both are emotions, yes they are opposite of each other but still they are emotions. So it highly depends on the world and its common usage in human natural language to determine its context.
# Here we found another example, (cold, cool, warm). The distance between cold and warm is smaller than cold and cool which are synomyms in this case, but the reason could be the corpus that is used, you see there are many cases where cool has other meanings than something cold, and many cases were warm and cold used in the same context.

# %% [markdown]
# ### Question 2.4: Analogies with Word Vectors [written] (1.5 points)
# Word vectors have been shown to *sometimes* exhibit the ability to solve analogies. 
# 
# As an example, for the analogy "man : grandfather :: woman : x" (read: man is to grandfather as woman is to x), what is x?
# 
# In the cell below, we show you how to use word vectors to find x using the `most_similar` function from the __[GenSim documentation](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.most_similar)__. The function finds words that are most similar to the words in the `positive` list and most dissimilar from the words in the `negative` list (while omitting the input words, which are often the most similar; see [this paper](https://www.aclweb.org/anthology/N18-2039.pdf)). The answer to the analogy will have the highest cosine similarity (largest returned numerical value).

# %%
# Run this cell to answer the analogy -- man : grandfather :: woman : x
pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'grandfather'], negative=['man']))

# %% [markdown]
# Let $m$, $g$, $w$, and $x$ denote the word vectors for `man`, `grandfather`, `woman`, and the answer, respectively. Using **only** vectors $m$, $g$, $w$, and the vector arithmetic operators $+$ and $-$ in your answer, what is the expression in which we are maximizing cosine similarity with $x$?
# 
# Hint: Recall that word vectors are simply multi-dimensional vectors that represent a word. It might help to draw out a 2D example using arbitrary locations of each vector. Where would `man` and `woman` lie in the coordinate plane relative to `grandfather` and the answer?

# %% [markdown]
# #### <font color="red">Write your answer here.</font>
# ## [Erfan:]
# 
# #### we want to maximize (g - m + w) similarity with x
# (Like King - Man + Woman ~= Queen example from the lecture) 

# %% [markdown]
# ### Question 2.5: Finding Analogies [code + written]  (1.5 points)
# Find an example of analogy that holds according to these vectors (i.e. the intended word is ranked top). In your solution please state the full analogy in the form x:y :: a:b. If you believe the analogy is complicated, explain why the analogy holds in one or two sentences.
# 
# **Note**: You may have to try many analogies to find one that works!

# %%
# ------------------
# Write your implementation here.
# analogy: England : British :: Iran : x
pprint.pprint(wv_from_bin.most_similar(positive=['iran', 'british'], negative=['england']))

# analogy: Pants : Man :: Skirt : x
pprint.pprint(wv_from_bin.most_similar(positive=['skirt', 'man'], negative=['pants']))

# analogy: Jobs : Apple :: Gates : x
pprint.pprint(wv_from_bin.most_similar(positive=['gates', 'apple'], negative=['jobs']))

# analogy: Bat : Batman :: Spider : x
pprint.pprint(wv_from_bin.most_similar(positive=['spider', 'batman'], negative=['bat']))

# ------------------

# %% [markdown]
# #### <font color="red">Write your answer here.</font>
# ## [Erfan:]
# - analogy: England : British :: Iran : x
# - expected answer: Iranian\Persian
# - analogy: Pants : Man :: Skirt : x
# - expected answer: Woman
# - analogy: Jobs : Apple :: Gates : x
# - expected answer: Microsoft
# - analogy: Bat : Batman :: Spider : x
# - expected answer: Spiderman
# 

# %% [markdown]
# ### Question 2.6: Incorrect Analogy [code + written] (1.5 points)
# Find an example of analogy that does *not* hold according to these vectors. In your solution, state the intended analogy in the form x:y :: a:b, and state the (incorrect) value of b according to the word vectors.

# %%
# ------------------
# Write your implementation here.

# analogy: Fire : Melting :: Ice : x
pprint.pprint(wv_from_bin.most_similar(positive=['ice', 'melting'], negative=['fire']))

# ------------------

# %% [markdown]
# #### <font color="red">Write your answer here.</font>
# ## [Erfan]:
# - analogy: Fire : Melting :: Ice : x
# - expected ans: Freezing,
# - generated ans: melt, glaciers, floes, ...

# %% [markdown]
# ### Question 2.7: Guided Analysis of Bias in Word Vectors [written] (1 point)
# 
# It's important to be cognizant of the biases (gender, race, sexual orientation etc.) implicit in our word embeddings. Bias can be dangerous because it can reinforce stereotypes through applications that employ these models.
# 
# Run the cell below, to examine (a) which terms are most similar to "girl" and "toy" and most dissimilar to "boy", and (b) which terms are most similar to "boy" and "toy" and most dissimilar to "girl". Point out the difference between the list of female-associated words and the list of male-associated words, and explain how it is reflecting gender bias.

# %%
# Run this cell
# Here `positive` indicates the list of words to be similar to and `negative` indicates the list of words to be
# most dissimilar from.
pprint.pprint(wv_from_bin.most_similar(positive=['girl', 'toy'], negative=['boy']))
print()
pprint.pprint(wv_from_bin.most_similar(positive=['boy', 'toy'], negative=['girl']))

# %% [markdown]
# #### <font color="red">Write your answer here.</font>
# ## [Erfan]:
# for girls it listed barbie, doll and ..., for boys it listed robot, pet and ...
# This list considered doll as girlish toy and robot as toy for boys. This is a bias that is made by our model, which is similar to the old boys and girls specific toys bias which is that girls should play with these specific toys and boys should be playing with other set of specific toys. And if a boy or a girl want to play with a toy from other set of toys, it is considered as anormal behaviour

# %% [markdown]
# ### Question 2.8: Independent Analysis of Bias in Word Vectors [code + written]  (1 point)
# 
# Use the `most_similar` function to find another case where some bias is exhibited by the vectors. Please briefly explain the example of bias that you discover.

# %%
# ------------------
# Write your implementation here.
a = 'parent' # parent
b = 'woman'
c = 'man'
pprint.pprint(wv_from_bin.most_similar(positive=[c, a], negative=[b]))
print()
pprint.pprint(wv_from_bin.most_similar(positive=[b, a], negative=[c]))

# ------------------

# %% [markdown]
# #### <font color="red">Write your answer here.</font>
# ## [Erfan]:
# Assigning company teminology to men parenting and "mother, pregnant, children" to women by the model, seems to be biased. This is simmilar to old gender bias of female staying at home and men do the outside works

# %% [markdown]
# ### Question 2.9: Thinking About Bias [written] (2 points)
# 
# Give one explanation of how bias gets into the word vectors. What is an experiment that you could do to test for or to measure this source of bias?

# %% [markdown]
# #### <font color="red">Write your answer here.</font>
# ## [Erfan]: 
# - I think the reason behind these biases are the resouce which was used to train these models. Models are not inherently biased, they get their bias from the resouce they consume. So to answer the quesion, We are the reason that our models are biased, we generated biased data and they used by models.

# %% [markdown]
# # <font color="blue"> Submission Instructions</font>
# 
# 1. Click the Save button at the top of the Jupyter Notebook.
# 2. Select Cell -> All Output -> Clear. This will clear all the outputs from all cells (but will keep the content of all cells). 
# 2. Select Cell -> Run All. This will run all the cells in order, and will take several minutes.
# 3. Once you've rerun everything, select File -> Download as -> PDF via LaTeX (If you have trouble using "PDF via LaTex", you can also save the webpage as pdf. <font color='blue'> Make sure all your solutions especially the coding parts are displayed in the pdf</font>, it's okay if the provided codes get cut off because lines are not wrapped in code cells).
# 4. Look at the PDF file and make sure all your solutions are there, displayed correctly. The PDF is the only thing your graders will see!
# 5. Submit your PDF on Gradescope.


