import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Galaxy:
    def __init__(self, dimension = 3, luminosity = 1, coords = np.array([0,0])):
        self.dimension = dimension
        self.coords = coords
        self.luminosity = luminosity