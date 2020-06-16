# KYC
Code for the KYC Paper

## Running Instructions
The instructions to train models and evaluate on custom data is similar for both tasks: Named Entity Recognition (`./NER/`) and Sentiment Analysis (`./sentiment`). In order to preprocess the data run     
 ``python process_data.py``   
 This script converts the raw data in the `data` directory and converts it to the required format along with some preprocessing like truncating sentences that exceed the limit, etc. It also computes the domain sketch as needed by the KYC model. The raw data is divided into directories `train` which contains the training data, `test` which contains unseen test data belonging to the clients that contribute to the training data and `ood` which contains out-of-distribution data from clients that provided no data when training. Each subdirectory consists of multiple text files - each of which corresponds to data from one client with the name of the file representing the name of the client. This can easily be replaced by custom data as needed.
 
 In order to train the model run   
 ``./train.sh
 ``   
 This will sequentially train the KYC model followed by the vanilla model on the data inside the `data/train` folder. The trained models will be dumped in the `models` directory.
 
 In order to evaluate (and compare) the performance of the 2 models run    
 ``./evaluate.sh``   
 This prints, for each of the models, the results to stdout as well as saves it in the `models` directory alongside the trained weights. It also dumps the predictions in the same directory.
 
 Note that for sentiment analysis, we use a validation set to determine how many epochs to train for. We train the model for a maximum of 5 epochs, storing predictions after every epoch and finally select the predictions from the epoch in which the validation accuracy was highest. This can be computed using the get_scores.py script (`python get_scores.py`).

# Todo
* Make requirements.txt
* Acknowledge those whose code is taken
* Add link to paper once made public
