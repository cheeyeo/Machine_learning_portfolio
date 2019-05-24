## Random character name generator

This project aims to create a random character level word generator in Keras based on
the [RNN effectiveness][RNN effectiveness article] article from Andrej Karpathy, which shows an example in
numpy and python.

I have rewritten it in Keras with the following changes:

* It reframes the problem as a seq2seq learning problem, hence requiring the
inputs and outputs to be reshaped to be 3D.

* Added a gradient clipping to the Adam optimizer


To train the model, run `python train.py`

To generate random text, run `python generate.py`

The generator will generate a single newline character in some cases and this will stop the generation process. Run the generator a few times to get the character of desired length.

## Todo

* Address issue of generating newlines to reach desired character length

* Train for more epochs and test accuracy.

* Make the generator accept a seed character


## References
[RNN effectiveness article](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
[RNN Source code](https://gist.github.com/karpathy/d4dee566867f8291f086)

## Useful hints for variable length RNN

* https://medium.freecodecamp.org/applied-introduction-to-lstms-for-text-generation-380158b29fb3

* https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras

* https://github.com/keras-team/keras/issues/3086
