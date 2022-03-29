#!/usr/bin/env python

# os libraries
import os
import argparse

# numerical libraries
import numpy as np

# dependencies
from data import *
from models import *

# parser initialisation
parser = argparse.ArgumentParser(description='Kernel Methods for Machine Learning Data Challenge 2022')

# training settings
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="Folder where train and test data is located (default: data).")
parser.add_argument('--mode', type=str, default='val', metavar='M',
                    help='Validation (val) or evaluation (eval) mode (default: eval).')
parser.add_argument('--split_size', type=float, default=.2, metavar='S',
                    help='Validation/training split size for validation mode (default: 0.2).')
parser.add_argument('--transform', type=str, default="histogram", metavar='T',
                    help='Data transform (default: histogram).')
parser.add_argument('--batch_size', type=int, default=10, metavar='B',
                    help='Number of batches for validation model (default: 10).')
parser.add_argument('--model', type=str, default='KFDA', metavar='MO',
                    help='Model to use for prediction (default: KFDA).')
parser.add_argument('--kernel', type=str, default='RBF', metavar='K',
                    help='Kernel to use for prediction (default: RBF).')

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
    data_loader = Loader(Xtr_path, Xte_path, Ytr_path, args.transform)

    # set classifier
    classifier = classifiers[args.model]

    # validation mode
    if args.mode == 'val':

        # initialise prediction scores
        scores = []

        print('-'*40)

        for batch in range(args.batch_size):

            # load train and validation data
            Xtr, Xval, Ytr, Yval = data_loader.load_train_val(args.split_size)

            # train classifier
            classifier.fit(Xtr, Ytr, args.kernel)

            # predict on valiation data
            Yval_pred = classifier.predict(Xval)

            # compute score
            score = (1/Yval.shape[0]) * np.sum(np.equal(Yval, Yval_pred))

            # get score for batch
            scores.append(score)

            # print current score
            print('Score for batch {}/{}: {:.2f}%'.format(batch+1, args.batch_size, 100*score))

        # print results
        print('-'*40)
        print('Model used: ' + args.model + ' | ' + 'Kernel used: ' + args.kernel)
        print('Mean prediction score: {:.2f}%'.format(100*np.mean(scores)))

    # evaluation mode
    elif args.mode == 'eval':

        # load full data
        Xtr, Xte, Ytr = data_loader.load_train_test()

        # train classifier
        classifier.fit(Xtr, Ytr, args.kernel)

        # predict on test data
        Yte = classifier.predict(Xte)

        # export results
        df = pd.DataFrame({'Prediction' : Yte})
        df.index += 1
        df.to_csv('submissions/Yte_pred.csv', index_label='Id')

        # print success
        print('Output successfully exported to: submissions/Yte_pred.csv!')

# run main function
if __name__ == '__main__':
    main()
