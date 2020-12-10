from scipy.stats import kstest


class KolmogorovSmirnov:

    def __init__(self, obs_array_1, obs_array_2):
        self.obs_array_1 = obs_array_1
        self.obs_array_2 = obs_array_2

    def perform_test(self):
        statistic, pvalue = kstest(rvs=self.obs_array_1, cdf=self.obs_array_2)
        print('Kolmogorov Smirnov test performed.')

        return statistic, pvalue