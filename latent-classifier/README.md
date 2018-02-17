# Latent Classifier

Images that looks a like are often misclassified as each other. We are developing a model to penalize misclassifications more based upon how similar their features are. 

1. Train an autoencoder to produce a latent representation of the inputs

2. Use the latent features produced from the encoder and train with the classifyer

3. Penalize the misclassifications inversely proportional to the distance (L1 or L2) of the features. 
  