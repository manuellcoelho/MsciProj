from Universe import Universe
import random
import numpy as np
import matplotlib.pyplot as plt

class EventGenerator(Universe):
    def __init__(self, galaxy_count = 1, dimension = 3, luminosity_gen_type = "Fixed",
                 coord_gen_type = "Random", spacing = 1,
                 event_count = 1, event_distribution = "Random",
                 noise_distribution = "gauss"):
        super().__init__(galaxy_count, dimension, luminosity_gen_type,
                 coord_gen_type, spacing)
        self.event_distribution = event_distribution
        self.event_count = event_count

        self.noise_distribution = noise_distribution
        self.noise_sigma = self.spacing/50

        self.BH_galaxies = np.empty((self.event_count), dtype = object)
        self.BH_coords = np.zeros((self.event_count, self.dimension))
        self.BH_luminosities = np.zeros((self.event_count))
        self.BH_detected_coords = np.empty((self.event_count, self.dimension))

        self.event_generator = dict({"Random": self.random_galaxy})
        self.coord_noise_generator = dict({"gauss": self.gauss_noise})
        self.generate_events(event_distribution = self.event_distribution,
                             noise_distribution = self.noise_distribution)

    def generate_events(self, event_distribution, noise_distribution):
        for i in range(self.event_count):
            selected = self.event_generator[event_distribution]()
            noise = self.coord_noise_generator[noise_distribution](sigma = self.noise_sigma)

            self.BH_galaxies[i] = self.galaxies[selected]
            self.BH_coords[i] = self.coords[selected]
            self.BH_detected_coords[i] = self.coords[selected] + noise
            self.BH_luminosities[i] = self.luminosities[selected]

    # def create_contour(self, center, error_prob = "gauss"):
    #     self.BH_detected_contour[i] =

    def random_galaxy(self):
        return random.randint(0, self.galaxy_count-1)

    def gauss_noise(self, sigma):
        return np.random.randn((self.dimension))*sigma

    def plot_universe_and_events(self, show = True):
        fig, ax = self.plot_universe(show = False)
        x, y = zip(*self.BH_coords)
        xhat, yhat = zip(*self.BH_detected_coords)
        for (x, y, s) in zip(x, y, self.BH_luminosities):
            ax.add_artist(plt.Circle(xy=(x, y), radius=s, color="r"))
        for (xhat, yhat, s) in zip(xhat, yhat, self.BH_luminosities):
            ax.add_artist(plt.Circle(xy=(xhat, yhat), radius=s, color="g"))
        if show:
            plt.show()



Gen = EventGenerator(event_count=1, spacing = 200, dimension = 2, galaxy_count=100,
                     luminosity_gen_type = "Random", coord_gen_type = "Random")

Gen.plot_universe_and_events()



