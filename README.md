<h1>Semi-Supervised-techniques (Active Learning and Self-Training)</h1>

<h3>Libraries needed:</h3>
<hr>
<h5>Numpy <br> Pandas <br> sklearn <br> NLTK <br> heapq <br> scipy </h5>

<h3> Sampling Folder: </h3>
<hr>
<h5>In Sampling folder we have two files: <br> <br> 1. filter.py <br> 2. sample.py <br> <br>
  In "filter.py" file we remove all reviews which are selected by sample.py from pool of unlabled data to remove repitition of reviews.  <br>
  In "sample.py" file we get sample from pool of unlabeled data. </h5>

<h3> Active_Learning_Techniques Folder: </h3>
<hr>
<h5>In Active_Learning_Techniques folder we have three files: <br> <br> 1. highentropy.py <br> 2. leastCP.py <br> 3. smallmargin.py <br> <br>
 We used three techniques of active learning <br>
 In "highentropy.py" file, we apply high entropy technique and generate ".csv" file by using selected reviews sentences of "sample.py" and put all selected reviews in "highentropy.csv"  <br>
 In "leastCP.py" (least confidence prediction) file, we apply least confidence prediction technique and generate ".csv" file by using selected reviews sentences of "sample.py" and put all selected reviews in "leastCP.csv"  <br> 
In "smallmargin.py" file, we apply smallmargin technique and generate ".csv" file by using selected reviews sentences of "sample.py" and put all selected reviews in "smallmargin.csv"  <br>
</h5>

<h3> Self_training Folder: </h3>
<hr>
<h5>In  Self_training folder we have only one files: <br> <br> 1. selftraining.py <br> <br>
 In "selftraining.py" file, we apply selftraining techniqueand generate ".csv" file by using selected reviews sentences of "sample.py" and put all selected reviews in  "selflearning.csv". <br> </h5>

<h3> Model_Evaluation Folder: </h3>
<hr>
<h5>In  Model_Evaluation folder we have only one files: <br> <br> 1. modelevaluation.py <br> <br>
In "modelevaluation.py" file, we evaluate our results.
</h5>
