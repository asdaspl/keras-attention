#!/usr/bin/env python3

"""
    Runs a simple Neural Machine Translation model
    Type `python run.py -h` for help with arguments.
"""
import os
import argparse

from keras.callbacks import ModelCheckpoint

from models.NMT import simpleNMT
from data.reader import Data, Vocabulary
from utils.metrics import all_acc
from utils.examples import run_examples

cp = ModelCheckpoint("./weights/NMT.{epoch:02d}-{val_loss:.2f}.hdf5",
                     monitor='val_loss',
                     verbose=0,
                     save_best_only=True,
                     save_weights_only=True,
                     mode='auto')

# create a directory if it doesn't already exist
if not os.path.exists('./weights'):
    os.makedirs('./weights/')


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Dataset functions
    input_vocab  = Vocabulary('./data/human_vocab.json', )
    output_vocab = Vocabulary('./data/machine_vocab.json')

    print('Loading datasets.')
    training   = Data(args.training_data,   input_vocab, output_vocab)
    validation = Data(args.validation_data, input_vocab, output_vocab)
    training  .load()
    validation.load()

    if args.paddings:
        paddings = tuple(int(i) for i in args.paddings.split(','))
        if len(paddings) == 1:
            paddings = paddings * 2
        assert len(paddings) == 2 and all(isinstance(i, int) for i in paddings)
    else:
        paddings = tuple(max(i) for i in zip(training.lean_paddings, validation.lean_paddings))

    training  .transform(paddings)
    validation.transform(paddings)
    print('Datasets Loaded.')

    print('Compiling Model.')
    model = simpleNMT(in_padding            = paddings[0],
                      out_padding           = paddings[1],
                      in_vocab_size         = input_vocab.size(),
                      out_vocab_size        = output_vocab.size(),
                      embedding_mul         = 7,
                      encoder_units         = 256,
                      decoder_units         = 256,
                      embedding_learnable   = False,
                      trainable             = True,
                      return_probabilities  = False)

    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', all_acc])
    print('Model Compiled.')
    print('Training. Ctrl+C to end early.')

    try:
        model.fit_generator(generator       =training.generator(args.batch_size),
                            steps_per_epoch =100,
                            validation_data =validation.generator(args.batch_size),
                            validation_steps=100,
                            callbacks       =[cp],
                            workers         =1,
                            verbose         =1,
                            epochs          =args.epochs)

    except KeyboardInterrupt as e:
        print('Model training stopped early.')

    print('Model training complete.')
    run_examples(model, input_vocab, output_vocab)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    named_args = parser.add_argument_group('named arguments')

    named_args.add_argument('-e', '--epochs', metavar='|',
                            help="""Number of Epochs to Run""",
                            required=False, default=50, type=int)

    named_args.add_argument('-g', '--gpu', metavar='|',
                            help="""GPU to use""",
                            required=False, default='0', type=str)

    named_args.add_argument('-p', '--paddings', metavar='|',
                            help="""fixed lengths of i/o sequence""",
                            required=False, default=None, type=str)

    named_args.add_argument('-t', '--training-data', metavar='|',
                            help="""Location of training data""",
                            required=False, default='./data/training.csv')

    named_args.add_argument('-v', '--validation-data', metavar='|',
                            help="""Location of validation data""",
                            required=False, default='./data/validation.csv')

    named_args.add_argument('-b', '--batch-size', metavar='|',
                            help="""Location of validation data""",
                            required=False, default=32, type=int)
    args = parser.parse_args()
    print(args)

    main(args)
