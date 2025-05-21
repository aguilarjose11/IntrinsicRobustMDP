#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:55:08 2025

@author: joseaguilar
"""

from collections.abc import Callable
from typing import Any
from torch import Tensor

#type Tensor = 

State  = Any
Action = Any

# \phi(s, a)
Phi = Callable[[State, Action], Tensor]
# \mu(s') -> 
Mu  = Callable[[State], Tensor]
# \pi(s) -> a
