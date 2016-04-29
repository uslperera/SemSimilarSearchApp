#!/usr/bin/python
# -*- coding: ascii -*-

__author__ = "Shamal Perera"
__copyright__ = "Copyright 2016, SemSimilar Project"
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "uslperera@gmail.com"


class InvalidProcessorCount(Exception):
    """Throws if the processor count given is more than the number of physical processors in the system"""

    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super(InvalidProcessorCount, self).__init__(message)
