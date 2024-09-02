# TextGenerator
Trains a Bidirectional LSTM on given source text to produce new text

## Use:
To train your own model and produce your own text, do the following:

1. Clone this repo.
2. Choose your source text. A source of about 5 paragraphs is a pretty good starting point (the longer the source, the longer the training).
3. Paste source text into an arbitrarily named file with a ".txt" extension into the "./source_text" directory.
4. Run the bash script, pass as an argument the name of the file you created (without directory, just the filename), and a seed sentence from which to generate.

The bash script will first clean your source text and place a cleaned version into "./source_text_cleaned" which will then be used to train the model. The model is configured to output all training epochs and generate a new text of size 200 (tokens) based off of the seed you enter.

