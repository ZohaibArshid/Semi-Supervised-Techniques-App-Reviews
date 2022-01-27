<h1>Semi-Supervised-techniques (Active Learning and Self-Training)</h1>

<h3>Libraries needed:</h3>
<h5>Numpy <br> Pandas <br> sklearn <br> NLTK <br> heapq <br> scipy </h5>

<h3> Sampling Folder: <//h3>
<h5>In Sampling folder we have two files: <br> <br> 1. filter.py <br> 2. sample.py <br> <br>
  In "filter.py" file we remove all reviews which are selected by sample.py from pool of unlabled data to remove repitition of reviews.  <br>
  In "sample.py" file we get sample from pool of unlabeled data.
   </h5>


<h6>"selftraining.py" file applying selftraining technique and put all seleted reviews in "selflearning.csv". 

"leastcp.py" file applying leastcp strategy and put all selected reviews in "leastcp.csv"

"smallmargin.py" file applying small margin strategy and put all selected reviews in "smallmargin.csv"

"highentropy.py" file applying high entropy strategy and put all selected reviews in "highentropy.csv"

"modelevaluation.py" file used for calculating results. </h6>
