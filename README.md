# Audio-Classification

# Problem Formulation

The problem that I am looking to solve is to build a machine learning pipeline that takes as an input a Potter or a StarWars audio segment and predicts its song label (either Harry or StarWars).

Both songs clearly sound different, however they do have similarities making feature extraction a key area to increase accuracy of the results. This is also made harder when the dataset is both hums and whistles, as they both have different features.

# Machine Learning Pipeline

Below is the machine learning pipeline:

1.   Data Input
2.   Transformation: Feature extraction and z-score normalisation
4.   Transformation: Feature Selection using a covariance matrix and an F test ('SelectKBest')
3.   Modelling using Logistic Regression and KNN and then tuning hyperparameters with K-fold cross validation.
4.   Testing on unseen data using the candidate models.

# Transformation Stage

The first step in the pipeline is transformation, namely feature extraction. Currently the files are complex with high dimensionality, so to make them more interpretable on a large scale we will extract seven features:


- Pitch standard devistion

- Pitch range

- Voiced probability - probability that a frame will be voiced

- Beats per minute (aim to capture the tempo of the song)

- Beat start: Captures the time between the first and nth beat

- Beat end: Captures the time between the nth last and last beat

- Fastest_quad: Splitting the song into 8 segments, and counting the number of beats for the fastest 2 segments. This tries to capture the beat at the fastest part of the song.

I realised that detecting a song would be less focused on certain metrics (say pitch centrality) as there are both hums and whistles for each song, however range a standard deviation of the pitch would privide some context. The main metrics would be around tempo and beat, so I studied the librosa documentation to understand which features I could use to focus on this.

After this I would go on to select three metrics by studying scatterplots, the covariance matrix and using an F test and the 'SelectKBest' Sklearn function to identify the 4 features to use in the model.


# Methodology

The results will be validated with k-fold cross validation, using accuracy. At this stage I will be tuning hyperparameters. After this, I will test each model (with the optimum hyperparameters) on an unseen test dataset.

I chose k=5 based off testing various values of K on the validation data as shown below.

![KNN]((https://user-images.githubusercontent.com/95233010/194070490-80debf7a-d5d7-42cc-a3d6-97e0bcf20958.png))

Test Accuracy: 75%





