from enum import Enum
from typing import Optional


class AfDefinition:
    """
    Define the activation function properties. Used by the builder to
    construct the appropriate activation function by its definition.
    """
    class AfType(Enum):
        """
        The activation function type
        """
        TRAD = 0
        """traditional (fixed) activation function"""

        ADA_AHAF = 1
        """Adaptive Hybrid Activation Function"""

        ADA_FUZZ = 2
        """Fuzzy activation function"""

        ADA_LEAF = 3
        """Learnable Extended Activation Function"""

    class AfInterval:
        """
        Stores the activation function domain for functions with limited
        domains (i.e. Fuzzy activation functions) as an interval between
        the minimum and the maximum input value.
        """
        def __init__(self, start: float, end: float, n_segments: int = 0):
            """
            :param start: the first value in the interval
            :param end: the last value in the interval
            :param n_segments: the number of piece-wise segments to cover the
                   input interval
            """
            self.start = start
            self.end = end
            self.n_segments = n_segments

    def __init__(
            self, af_base: str = "ReLU", af_type: AfType = AfType.TRAD,
            af_interval: Optional[AfInterval] = None,
            af_sharing: bool = False
    ):
        """
        :param af_base: the base activation function, for adaptive functions -
               determines the initial activation function parameters, for fixed
               activations - determines the activation function to use
        :param af_type: the activation function type (traditional i.e. fixed,
               AHAF, Fuzzy activation, LEAF)
        :param af_interval: for Fuzzy activations - defines the activation
               function domain, ignored for all other activations (set to None)
        :param af_sharing: True to use the same activation function instance for
               all neurons in this layer, False by default
        """
        self.af_base = af_base
        self.af_type = af_type
        self.interval = af_interval
        self.af_sharing = af_sharing
