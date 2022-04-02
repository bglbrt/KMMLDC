#!/usr/bin/env python

# os and file management libraries
import os
import argparse
import importlib

# numerical libraries
import numpy as np

# dependencies
from data import *
from models import *

# parser initialisation
parser = argparse.ArgumentParser(description='Kernel Methods for Machine Learning Data Challenge 2022')

# training settings
parser.add_argument('--data', type=str, default='data.nosync', metavar='D',
                    help="Folder where train and test data is located (default: data).")
parser.add_argument('--mode', type=str, default='val', metavar='M',
                    help='Validation (val) or evaluation (eval) mode (default: eval).')
parser.add_argument('--split_size', type=float, default=.2, metavar='S',
                    help='Validation/training split size for validation mode (default: 0.2).')
parser.add_argument('--augment', type=str, default="all", metavar='A',
                    help='Data augmentation (default: horizontal).')
parser.add_argument('--angle', type=float, default=15, metavar='AN',
                    help='Angle for rotate data augmentation in degrees (default: 15).')
parser.add_argument('--transform', type=str, default="hog", metavar='T',
                    help='Data transform (default: hog).')
parser.add_argument('--cells_size', type=int, default=4, metavar='CS',
                    help='Cells size for HOG data transform (default: 4).')
parser.add_argument('--n_orientations', type=int, default=8, metavar='NO',
                    help='Number of gradient orientations for HOG data transform (default: 8).')
parser.add_argument('--batch_size', type=int, default=10, metavar='B',
                    help='Number of batches for validation model (default: 10).')
parser.add_argument('--decomposition', type=str, default=None, metavar='DCP',
                    help='Decomposition to use dimensionnality reduction (default: None).')
parser.add_argument('--model', type=str, default='KFDA', metavar='MO',
                    help='Model to use for prediction (default: KFDA).')
parser.add_argument('--KPCA_n_components', type=int, default=200, metavar='KPCANC',
                    help='Kernel Principal Component Analysis number of components (default: 50).')
parser.add_argument('--KRR_gamma', type=float, default=1e-6, metavar='KRRG',
                    help='Kernel Ridge Regression classifier regularization parameter (default: 1e-6).')
parser.add_argument('--KFDA_gamma', type=float, default=1e-6, metavar='KFDAG',
                    help='Kernel Fisher Discriminant Analysis regularization parameter (default: 1e-6).')
parser.add_argument('--KFDA_n_components', type=int, default=50, metavar='KFDANC',
                    help='Kernel Fisher Discriminant Analysis number of components (default: 50).')
parser.add_argument('--KSVC_C', type=float, default=1.0, metavar='KSVCC',
                    help='Kernel Support Vector Classifier regularization parameter (default: 1.0).')
parser.add_argument('--KSVC_decision_function', type=str, default='ovo', metavar='KSVCDF',
                    help='Kernel Support Vector Classifier decision function (default: "ovo").')                   
parser.add_argument('--kernel', type=str, default='RBF', metavar='K',
                    help='Kernel to use for prediction (default: RBF).')
parser.add_argument('--decomposition_kernel', type=str, default='RBF', metavar='K',
                    help='Kernel to use for prediction (default: RBF).')
parser.add_argument('--Polynomial_a', type=float, default=1., metavar='POLYA',
                    help='Affine parameter for the polynomial kernel (default: 1.).')
parser.add_argument('--Polynomial_c', type=float, default=1., metavar='POLYC',
                    help='Bias parameter for the polynomial kernel (default: 1.).')
parser.add_argument('--Polynomial_d', type=int, default=1, metavar='POLYD',
                    help='Power parameter for the polynomial kernel (default: 1).')
parser.add_argument('--RBF_sigma', type=float, default=1., metavar='RBFSIG',
                    help='Bandwith parameter for the RBF kernel (default: 1.).')
parser.add_argument('--Exponential_sigma', type=float, default=1., metavar='EXPSIG',
                    help='Bandwith parameter for the Exponential kernel (default: 1.).')
parser.add_argument('--Laplacian_sigma', type=float, default=1., metavar='LAPSIG',
                    help='Bandwith parameter for the Laplacian kernel (default: 1.).')
parser.add_argument('--ANOVA_sigma', type=float, default=1., metavar='ANOVASIG',
                    help='Bandwith parameter for the ANOVA kernel (default: 1.).')
parser.add_argument('--ANOVA_d', type=int, default=1, metavar='ANOVAD',
                    help='Power parameter for the ANOVA kernel (default: 1).')
parser.add_argument('--TanH_a', type=float, default=1., metavar='TANHA',
                    help='Affine parameter for the Hyperbolic Tangent kernel (default: 1.).')
parser.add_argument('--TanH_c', type=float, default=1., metavar='TANHC',
                    help='Bias parameter for the Hyperbolic Tangent kernel (default: 1.).')
parser.add_argument('--RationalQuadratic_c', type=float, default=1., metavar='RQC',
                    help='Regularization parameter for the Rational Quadratic kernel (default: 1.).')
parser.add_argument('--Multiquadratic_c', type=float, default=1., metavar='MQC',
                    help='Regularization parameter for the Multiquadratic kernel (default: 1.).')
parser.add_argument('--InverseMultiquadratic_c', type=float, default=1., metavar='IMQC',
                    help='Regularization parameter for the Inverse Multiquadratic kernel (default: 1.).')
parser.add_argument('--Wave_c', type=float, default=1., metavar='WC',
                    help='Regularization parameter for the Wave kernel (default: 1.).')
parser.add_argument('--Wave_theta', type=float, default=np.pi/2, metavar='WTHETA',
                    help='Angle parameter for the Wave kernel (default: np.pi/2).')
parser.add_argument('--Power_d', type=int, default=1, metavar='PD',
                    help='Power parameter for the Power kernel (default: 1).')
parser.add_argument('--Log_d', type=int, default=1, metavar='LOGD',
                    help='Power parameter for the Log kernel (default: 1).')
parser.add_argument('--Cauchy_sigma', type=float, default=1., metavar='CAUCHYSIG',
                    help='Variance parameter for the Cauchy kernel (default: 1.).')
parser.add_argument('--ChiSquare_sigma', type=float, default=1., metavar='CHI2SIG',
                    help='Variance parameter for the Chi Square kernel (default: 1.).')

# main function
def main():
    '''
    Main function for validating and evaluating models.
    '''

    # parse arguments for training settings
    args = parser.parse_args()

    # set kernel arguments
    if args.kernel == 'Linear':
        kernel_kwargs = {}

    elif args.kernel == 'Polynomial':
        kernel_kwargs = {'a': args.Polynomial_a,
                         'c': args.Polynomial_c,
                         'd': args.Polynomial_d}

    elif args.kernel == 'RBF':
        kernel_kwargs = {'sigma': args.RBF_sigma}

    elif args.kernel == 'Exponential':
        kernel_kwargs = {'sigma': args.Exponential_sigma}

    elif args.kernel == 'Laplacian':
        kernel_kwargs = {'sigma': args.Laplacian_sigma}

    elif args.kernel == 'ANOVA':
        kernel_kwargs = {'sigma': args.ANOVA_sigma,
                         'd': args.ANOVA_d}

    elif args.kernel == 'TanH':
        kernel_kwargs = {'a': args.TanH_a,
                         'c': args.TanH_c}

    elif args.kernel == 'RationalQuadratic':
        kernel_kwargs = {'c': args.RationalQuadratic_c}

    elif args.kernel == 'Multiquadratic':
        kernel_kwargs = {'c': args.Multiquadratic_c}

    elif args.kernel == 'InverseMultiquadratic':
        kernel_kwargs = {'c': args.InverseMultiquadratic_c}

    elif args.kernel == 'Wave':
        kernel_kwargs = {'c': args.Wave_c,
                         'theta': args.Wave_theta}

    elif args.kernel == 'Power':
        kernel_kwargs = {'d': args.Power_d}

    elif args.kernel == 'Log':
        kernel_kwargs = {'d': args.Log_d}

    elif args.kernel == 'Cauchy':
        kernel_kwargs = {'sigma': args.Cauchy_sigma}

    elif args.kernel == 'ChiSquare':
        kernel_kwargs = {'sigma': args.ChiSquare_sigma}

    elif args.kernel == 'HistogramIntersection':
        kernel_kwargs = {}

    else:
        raise NotImplementedError("Kernel not implemented!")

    # set kernel arguments
    if args.decomposition_kernel == 'Linear':
        decomp_kernel_kwargs = {}

    elif args.decomposition_kernel == 'Polynomial':
        decomp_kernel_kwargs = {'a': args.Polynomial_a,
                         'c': args.Polynomial_c,
                         'd': args.Polynomial_d}

    elif args.decomposition_kernel == 'RBF':
        decomp_kernel_kwargs = {'sigma': args.RBF_sigma}

    elif args.decomposition_kernel == 'Exponential':
        decomp_kernel_kwargs = {'sigma': args.Exponential_sigma}

    elif args.decomposition_kernel == 'Laplacian':
        decomp_kernel_kwargs = {'sigma': args.Laplacian_sigma}

    elif args.decomposition_kernel == 'ANOVA':
        decomp_kernel_kwargs = {'sigma': args.ANOVA_sigma,
                         'd': args.ANOVA_d}

    elif args.decomposition_kernel == 'TanH':
        decomp_kernel_kwargs = {'a': args.TanH_a,
                         'c': args.TanH_c}

    elif args.decomposition_kernel == 'RationalQuadratic':
        decomp_kernel_kwargs = {'c': args.RationalQuadratic_c}

    elif args.decomposition_kernel == 'Multiquadratic':
        decomp_kernel_kwargs = {'c': args.Multiquadratic_c}

    elif args.decomposition_kernel == 'InverseMultiquadratic':
        decomp_kernel_kwargs = {'c': args.InverseMultiquadratic_c}

    elif args.decomposition_kernel == 'Wave':
        decomp_kernel_kwargs = {'c': args.Wave_c,
                         'theta': args.Wave_theta}

    elif args.decomposition_kernel == 'Power':
        decomp_kernel_kwargs = {'d': args.Power_d}

    elif args.decomposition_kernel == 'Log':
        decomp_kernel_kwargs = {'d': args.Log_d}

    elif args.decomposition_kernel == 'Cauchy':
        decomp_kernel_kwargs = {'sigma': args.Cauchy_sigma}

    elif args.decomposition_kernel == 'ChiSquare':
        decomp_kernel_kwargs = {'sigma': args.ChiSquare_sigma}

    elif args.decomposition_kernel == 'HistogramIntersection':
        decomp_kernel_kwargs = {}

    else:
        raise NotImplementedError("Kernel not implemented!")


    # set decomposition arguments
    if args.decomposition:
        if args.decomposition == 'KPCA':
            decomposition_kwargs = {'kernel': args.decomposition_kernel,
                                    'n_components': args.KPCA_n_components,
                                    'decomp_kernel_kwargs': decomp_kernel_kwargs}
    
        else:
            raise NotImplementedError("Decomposition not implemented!")


    # set classifier arguments
    if args.model == 'KRR':
        classifier_kwargs = {'kernel': args.kernel,
                             'gamma': args.KRR_gamma,
                             'kernel_kwargs':kernel_kwargs}

    elif args.model == 'KFDA':
        classifier_kwargs = {'kernel': args.kernel,
                             'gamma': args.KFDA_gamma,
                             'n_components': args.KFDA_n_components,
                             'kernel_kwargs':kernel_kwargs}

    elif args.model == 'SVCC':
        classifier_kwargs = {'kernel': args.kernel,
                             'C':1.0,
                             'kernel_kwargs':kernel_kwargs}

    elif args.model == 'KSVC':
        classifier_kwargs = {'kernel': args.kernel,
                             'C': args.KSVC_C,
                             'decision_function': args.KSVC_decision_function,
                             'kernel_kwargs':kernel_kwargs}

    elif args.model == 'skSVC':
        classifier_kwargs = {'kernel': args.kernel,
                             'C': args.KSVC_C,
                             'decision_function': args.KSVC_decision_function,
                             'kernel_kwargs':kernel_kwargs}

    else:
        raise NotImplementedError("Model not implemented!")

    print('-'*40)
    print('Loading and pre-processing data...')

    Xtr_path = os.path.join(args.data, 'Xtr.csv')
    Xte_path = os.path.join(args.data, 'Xte.csv')
    Ytr_path = os.path.join(args.data, 'Ytr.csv')

    # declare loader
    data_loader = Loader(Xtr_path, Xte_path, Ytr_path, args.augment, args.angle, args.transform, args.cells_size, args.n_orientations)

    print('Data successfully loaded!')

    if args.decomposition:
        # set decomposition
        decomposition_class = getattr(importlib.import_module('models'+'.'+args.decomposition), args.decomposition)
        decomposition = decomposition_class()

    # set classifier
    classifier_class = getattr(importlib.import_module('models'+'.'+args.model), args.model)
    classifier = classifier_class()

    # validation mode
    if args.mode == 'val':

        # initialise prediction scores
        scores = []

        print('-'*40)

        for batch in range(args.batch_size):

            # load train and validation data
            Xtr, Xval, Ytr, Yval = data_loader.load_train_val(args.split_size)

            if args.decomposition:
                # reduce dimensions
                decomposition.fit(Xtr, Ytr, **decomposition_kwargs)
                Xtr = decomposition.predict(Xtr)
                Xval = decomposition.predict(Xval)
            
            # train classifier
            classifier.fit(Xtr, Ytr, **classifier_kwargs)

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

        print('-'*40)

        if args.decomposition:
                # reduce dimensions
                decomposition.fit(Xtr, Ytr, **decomposition_kwargs)
                Xtr = decomposition.predict(Xtr)
                Xval = decomposition.predict(Xval)

        # train classifier
        classifier.fit(Xtr, Ytr, **classifier_kwargs)

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
