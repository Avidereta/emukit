import scipy
import numpy as np
from scipy.stats import norm
from scipy.optimize import bisect

from emukit.core import InformationSourceParameter
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel
from emukit.core.parameter_space import ParameterSpace
from emukit.experimental_design.model_free.latin_design import LatinDesign
from emukit.experimental_design.model_free.random_design import RandomDesign

class MinValueEntropySearch(Acquisition):

    def __init__(self, model: IModel, space: ParameterSpace,
                 num_min_samples: int = 100, gridsize=10000) -> None:
        """
        Min-value Entropy search acquision function that aims to solve minimization problem.
        See the paper for mor details:

        :param model: GP model to compute the distribution of the minimum dubbed pmin.
        :param space: Domain space which we need for the sampling of the representer points
        :param num_min_samples: number of objective minima in sum approximation
        :param gridsize: number of samples to draw from the model in Gumbel sampling
        """
        super().__init__()

        self.model = model
        self.gridsize = gridsize
        self.num_min_samples = num_min_samples
        self._space = space
        self.mins = self.sample_minima()

    def evaluate(self, x):
        """
        Computes the information gain
        The exact formula for computations is based on the Eq.(6) of the max-value entropy search paper
        and adapted for the minimization problem
        """
        fmean, fvar = self.model.predict(x)
        norm = scipy.stats.norm(0.0, 1.0)
        #         self.mins = self.sample_minima()
        gamma = (np.expand_dims(self.mins, axis=0) - fmean) / np.sqrt(fvar)

        return np.sum(- gamma * norm.pdf(gamma) / (2. * (1 - norm.cdf(gamma))) - np.log(1 - norm.cdf(gamma)),
                      axis=1, keepdims=True) / self.num_min_samples

    def sample_minima(self):
        """
        Apply Gumbel sampling
        """
        N = np.shape(self.model.X)[0]
        Xrand = LatinDesign(self._space).get_samples(self.gridsize)
        fmean, fvar = self.model.predict(np.vstack((self.model.X, Xrand)))
        idx = np.argmin(fmean[:N])
        right = fmean[idx].flatten()  # + 2*np.sqrt(fvar[idx]).flatten()
        left = right

        probf = lambda x: np.exp(np.sum(norm.logcdf(-(x - fmean) / np.sqrt(fvar)), axis=0))

        i = 0
        while probf(left) < 0.75:
            left = 2. ** i * np.min(fmean - 5. * np.sqrt(fvar)) + (1. - 2. ** i) * right
            i += 1

        # Binary search for 3 percentiles
        q1, med, q2 = map(lambda val: bisect(lambda x: probf(x) - val, left, right, maxiter=10000, xtol=0.01),
                          [0.25, 0.5, 0.75])
        beta = (q1 - q2) / (np.log(np.log(4. / 3.)) - np.log(np.log(4.)))
        alpha = med + beta * np.log(np.log(2.))

        # obtain samples from y*
        mins = np.log(-np.log(1 - np.random.rand(self.num_min_samples).astype(np.float32))) * beta + alpha
        return mins

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return False


class MultiFidelityMinValueEntropySearch(MinValueEntropySearch):
    """
    Min-Value Entropy search acquisition for multi-fidelity (multi-information source) problems where the objective function is the output of one
    of the information sources. The other information sources provide auxiliary information about the objective function
    """

    def __init__(self, model: IModel, space: ParameterSpace,
                 target_information_source_index: int = None,
                 num_min_samples: int = 100, gridsize=10000) -> None:

        """

        :param model: GP model to compute the distribution of the minimum dubbed pmin.
        :param space: Domain space which we need for the sampling of the representer points
        :param sampler: mcmc sampler for representer points
        :param num_min_samples: integer determining how many in samples of objective minima to draw

        """

        self.model = model
        self.gridsize = gridsize
        self.num_min_samples = num_min_samples
        self._space = space

        # Find information source parameter in parameter space
        info_source_parameter, source_idx = _find_source_parameter(space)
        self.source_idx = source_idx

        # Assume we are in a multi-fidelity setting and the highest index is the highest fidelity
        if target_information_source_index is None:
            target_information_source_index = max(info_source_parameter.domain)
        self.target_information_source_index = target_information_source_index

        # Sampler of representer points should sample x location at the target information source only so make a
        # parameter space without the information source parameter
        parameters_without_info_source = space.parameters.copy()  # optimization space
        parameters_without_info_source.remove(info_source_parameter)
        self._space_without_info_source = ParameterSpace(parameters_without_info_source)

        self.mins = self.sample_minima()

    def evaluate(self, x):
        """
        Computes the information gain
        TODO: explain
        """
        fmean, fvar = self.model.predict(x)
        norm = scipy.stats.norm(0.0, 1.0)
        #         self.mins = self.sample_minima()
        #         gamma = (np.expand_dims(self.mins, axis=0) - 2*fmean - 3*fvar) / np.sqrt(fvar)
        gamma = (np.expand_dims(self.mins, axis=0) - fmean) / np.sqrt(fvar)

        return np.sum(- gamma * norm.pdf(gamma) / (2. * (1 - norm.cdf(gamma))) - np.log(1 - norm.cdf(gamma)),
                      axis=1, keepdims=True) / self.num_min_samples

    def sample_minima(self):
        """
        Apply Gumbel sampling
        TODO: explain
        """
        model_X_target = self.model.X[model.X[:, 1] == self.target_information_source_index]
        N = np.shape(model_X_target)[0]
        Xrand = LatinDesign(self._space_without_info_source).get_samples(self.gridsize)
        Xrand = np.array([[xi, self.target_information_source_index] for xi in Xrand])
        fmean, fvar = self.model.predict(np.vstack((model_X_target, Xrand)))
        idx = np.argmin(fmean[:N])
        right = fmean[idx].flatten()  # + 2*np.sqrt(fvar[idx]).flatten()
        left = right

        probf = lambda x: np.exp(np.sum(norm.logcdf(-(x - fmean) / np.sqrt(fvar)), axis=0))

        i = 0
        while probf(left) < 0.75:
            left = 2. ** i * np.min(fmean - 5. * np.sqrt(fvar)) + (1. - 2. ** i) * right
            i += 1

        # Binary search for 3 percentiles
        q1, med, q2 = map(lambda val: bisect(lambda x: probf(x) - val, left, right, maxiter=10000, xtol=0.01),
                          [0.25, 0.5, 0.75])
        beta = (q1 - q2) / (np.log(np.log(4. / 3.)) - np.log(np.log(4.)))
        alpha = med + beta * np.log(np.log(2.))

        # obtain samples from y*
        mins = np.log(-np.log(1 - np.random.rand(self.num_min_samples).astype(np.float32))) * beta + alpha
        return mins

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return False


def _find_source_parameter(space):
    # Find information source parameter in parameter space
    info_source_parameter = None
    source_idx = None
    for i, param in enumerate(space.parameters):
        if isinstance(param, InformationSourceParameter):
            info_source_parameter = param
            source_idx = i

    if info_source_parameter is None:
        raise ValueError('No information source parameter found in the parameter space')

    return info_source_parameter, source_idx