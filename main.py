#!/usr/bin/env python

# os libraries
import os
import argparse

# numerical libraries
import numpy as np

# dependencies
from data import *

# parser initialisation
parser = argparse.ArgumentParser(description='Kernel Methods for Machine Learning Data Challenge 2022')

# training settings
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="Folder where train and test data is located (default: data).")
parser.add_argument('--mode', type=str, default='eval', metavar='M',
                    help='Validation (val) or evaluation (eval) mode (default: eval).')
parser.add_argument('--split_size', type=float, default=0.2, metavar='S',
                    help='Validation/training split size for validation mode (default: 0.2).')
parser.add_argument('--batch_size', type=int, default=10, metavar='B',
                    help='Number of batches for validation model (default: 10).')
parser.add_argument('--model', type=str, default='KRR', metavar='MO',
                    help='Model to use for prediction (default: KRR).')
parser.add_argument('--kernel', type=str, default='linear', metavar='K',
                    help='Kernel to use for prediction (default: linear).')

# main function
def main():
    '''
    Main function for validating and evaluating models.
    '''

    # parse arguments for training settings
    args = parser.parse_args()

    Xtr_path = os.path.join(args.data, 'Xtr.csv')
    Xte_path = os.path.join(args.data, 'Xte.csv')
    Ytr_path = os.path.join(args.data, 'Ytr.csv')

    # declare loader
    data_loader = LOADER(Xtr_path, Xte_path, Ytr_path)

    # validation mode
    if args.mode == 'val':

        # initialise prediction scores
        scores = []

        for batch in args.batch_size:

            # load train and validation data
            Xtr, Xval, Ytr, Yval = data_loader.load_train_val(split_size=args.split_size)

            # train classifier
            classifier.train(Ytr, Xtr)

            # predict on valiation data
            Yval_pred = classifier.fit(Xval)

            # get score for batch
            scores.append(np.sum(np.equal(Yval, Yval_pred)))

        # print results
        print('Model used: ' + args.model + ' | ' + 'Kernel used: ' + args.kernel)
        print('Mean prediction score: {:.2f}'.format(np.mean(scores)))

    # evaluation mode
    elif args.mode == 'eval':

        # load full data
        Xtr, Xte, Ytr = data_loader.load_train_test()

        # train classifier
        classifier.train(Ytr, Xtr)

        # predict on test data
        Yte = classifier.fit(Xte)

        # export results
        df = pd.DataFrame({'Prediction' : Yte})
        df.index += 1
        dataframe.to_csv('submissions/Yte_pred.csv', index_label='Id')

        # print success
        print('Output successfully exported to: submissions/Yte_pred.csv!')

# run main function
if __name__ == '__main__':
    main()
