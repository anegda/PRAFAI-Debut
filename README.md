# PRAFAI-Debut

This github repository contains all the scripts regarding the experimentation for the Atrial Fibrillation (AF) Debut episode detection. This experimentation uses discharge reports for detecting the first episode of AF of a patient.

This article and repository are part of the PRAFAI project (Predicting Recurrence of Atrial Fibrillation using Artificial Intelligence).

## Scripts
The structure of the repository is the following:

    ├── Flair                           # Folder containing all the scripts for the training and evaluation of the Flair models.
    |   ├── Flair_Evaluate.py           # Script for training the Flair model.
    |   └── Flair_Train.py              # Script for evaluating the Flair model.
    |
    ├── feedforward_embeddings.py       # Script to train the feedforward model using embeddings for vectorizing.
    |
    ├── feedforward_tfidf.py            # Script to train the feedforward model using tf-idf for vectorizing.
    |
    ├── regular_expressions.json        # JSON with the regular expressions used for the vector generation process.
    |
    └── train_ml_classifier.py          # Script for training the LMs and to apply sliding-window and hyperparameter optmization.  
