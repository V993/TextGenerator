# TextGenerator
Deep Learning models trained on your chosen source text to produce new text based off of said source.

## Use:
This project has been optimized for easy use. To train your own model and produce your own text, do the following:

1. Clone this repo.
2. Choose your source text. A source of about 5 paragraphs is a pretty good starting point (the longer the source, the longer the training).
3. Paste source text into an arbitrarily named file with a ".txt" extension into the "./source_text" directory.
4. Run the bash script, pass as an argument the name of the file you created (without directory, just the filename), and a seed sentence from which to generate.
5. Enjoy your results.

The bash script will first clean your source text and place a cleaned version into "./source_text_cleaned" which will then be used to train the model. The model is configured to output all training epochs and generate a new text of size 200 (words) based off of the seed you enter.

## Q&A:

#### What is an epoch?
- An epoch is a full pass of the deep neural net through the input text you provided. The number of epochs often correlates to how much is learned, the model learns and corrects on each epoch.

#### What is a seed and what should it look like?
- A seed is an arbitrary sentence off of which the model will generate text. It should ideally be somehow related to your source text. E.g. if your source text is Republican speech data on the 2nd amendment (rep_speech.txt in "./source_text/"), your seed should ideally be related: "The second amendment is"
- The sentence can be saved to a file and used instead, but it doesn't matter. Ideally enclosing it in "" or '' is preferred. 
