# uncertainty-estimates
A project to find out a viable approach for quantifying the uncertainty of the model predictions.

## Step-1: Literature Survey
ref1: [Safer Classification by Synthesis](https://arxiv.org/abs/1711.08534)

`Synthesis`

- The `probability values` of Discriminative classifiers can be high even when the input is coming from out-of-sample distribution. This can be quantified by plotting `coverage` vs `risk` (by varying the classification threshold). Ideally when sufficiently large threshold is chosen, misclassification-rate should be zero.
- **IDEA:Generative Classification** Since discrimative classification can be seen as finding the projecting of the data onto a manifold such that seperating hyperplanes neatly segregate the data. Now, we can see how even for an out-of-distribution sample, we can have high-confidence prediction. Thus, We can try to model the `distance from the boundary` directly to quantify the confidence. 
- Let's say we have a class-conditional generative model  **M**. For a given test image `x`, find an image from each class that is closest(some distance-metric like l2) to it via the generative model, and label it's class to be the one that has highest similarity(least distance). Here we can use a threshold to determine if we can assign a class label or not. Here we might be able to avoid the below case.
- **Conclusion**: Use the generative model for thresholding. Use a CNN to predict the labels on these images. This gives us both the accuracy and shields us from making predictions on out-of-sample data-points.

 <img width="578" alt="Screen Shot 2021-12-27 at 3 29 17 PM" src="https://user-images.githubusercontent.com/21222766/147505241-a0cf3c76-dd6d-4fda-8563-ad2ccd4b9386.png">

`Implementation concerns`

- Monte-carlo sampling and L-BFGS is needed to find the optimal image for each class.

`Missing Pieces/Think on these things`

- Is the additional complexity warranted ? Can we use simpler approaches ?
- Generating images/samples is not needed for the task at hand. We can fit the generative model on a lower dimensional projection vectors.
- Decoupling of objectives is typically a bad idea in deep learning. Can we frame a sigle network/objective to perform both novelty detection and classification ? Note that the paper is quite old.
- Find some real-world data with `data-shift`,to ultimately bench mark your approach. Also think/quantify exactly what do you mean by `out-of-distribution` and `uncertainity`(epistemic vs aleatory uncertainty). Bayesian NN's seem to be a direct candidate here.

**NOTES**

`On Simpler Approaches`
- Train an ensemble of models(can be done within a single model with multiple heads) and use the `std` as a threshold to decide for `novelty` and the `avg` as the class label.



# Tools
We will be using the following tools for implementation.

[Pytorch-lightning](https://devblog.pytorchlightning.ai/lightning-tutorials-in-collaboration-with-the-university-of-amsterdam-uva-2499eaa0caad)

[Weights-Biases](https://wandb.ai/site)

[Streamlit](https://streamlit.io/)
