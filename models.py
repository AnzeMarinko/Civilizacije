# generate points distributed by model to draw distribution of L(N)
from numpy.random import uniform, normal, lognormal
from numpy import log10, roots, abs as nabs, exp as nexp

# all possible distributions and models (lists for iterating)
distributions = ["loguniform", "uniform", "halfgauss", "lognormal", "fixed"]  # all possible distributions
models = [1, 2, 3]


def sample_value(fromv, tov, distribution="fixed"):
    # random value from "distribution" distribution from "10**fromv" to "10**tov"
    if distribution == "loguniform":
        return uniform(fromv, tov)  # loguniform from [10**fromv, 10**tov]
    elif distribution == "uniform":
        return log10(uniform(10 ** fromv, 10 ** tov))  # uniform from [fromv, tov]
    elif distribution == "halfgauss":
        sigma_half_gauss = (10 ** tov - 10 ** fromv) / 3  # divided by 3 so that 3*sigma expands along whole interval
        return log10(nabs(normal(0, sigma_half_gauss)) + 10 ** fromv)  # gauss
    elif distribution == "lognormal":
        mean = (tov + fromv) / 2  # half of interval
        sigma = (tov - mean) / 3  # divided by 3 so that 3*sigma expands along whole interval
        return normal(mean, sigma)  # lognormal
    return tov  # if distribution=="fixed"


def life_dist(mean=0, sigma=50):
    # Sandberg: We used a lognormal distribution for the life emergence rate (log lambda ~ N(0,50)) and then
    # transformed it into a probability as fLife = 1-exp(-lambda).
    result = 0
    while result == 0:
        result = 1 - nexp(- lognormal(mean, sigma))
    return result


# get log(L) for max_n at random other parameters
# ============ Models ====================


def get_point_model_1(max_n=10, distribution=(0, 0, 0, 0, 0, 0)):
    # sample parameters in logarithmic scale
    RStarSample = sample_value(0, 2, distributions[distribution[0]])  # rate of new star born
    fPlanets = sample_value(-1, 0, distributions[distribution[1]])  # probability that star has planets
    nEnvironment = sample_value(-1, 0, distributions[distribution[2]])  # probability that it is earth-like
    fIntelligence = sample_value(-3, 0, distributions[distribution[3]])  # prob. some intelligent beings start to exist
    fCivilization = sample_value(-2, 0, distributions[distribution[4]])  # prob. this beings are possible to communicate
    #       with other planets

    logN = sample_value(0, log10(max_n), distributions[distribution[5]])
    fLife = life_dist(mean=0, sigma=50)    # probability that life begins
    fLifeEks = log10(fLife)

    # N = RStarSample + fPlanets + nEnvironment + fLifeEks + fInteligence + fCivilization + L
    logL = logN - (RStarSample + fPlanets + nEnvironment + fLifeEks + fIntelligence + fCivilization)
    return logL  # rate of birth of new civilisation


def get_point_model_2(max_n=10, distribution=(0, 0, 0)):
    # Model 2: https://arxiv.org/ftp/arxiv/papers/1510/1510.08837.pdf
    # Using Sandberg distribution
    # Model 2 opposite to original Drake equation - model 1 (calculates how many intelligent civilisations are out
    # there now), this model calculates how many intelligent civilisations were ever out there
    # it does not use L variable and uses total no. of stars in the universe instead of the rate at which the stars
    # are apperaing
    # for less known parameters: biotechnicalProbability, they calculated minimal threshold and uses different values
    astrophysicsProbability = normal(0.155, 0.73)  # distribution similar to the one calculated by our self
    if astrophysicsProbability > 2:  # so that the distribution look more similar
        astrophysicsProbability = sample_value(1.65, 1.95, distributions[distribution[0]])
    if astrophysicsProbability < -2:
        astrophysicsProbability = sample_value(-1.95, -1.65, distributions[distribution[0]])

    N = sample_value(0, log10(max_n), distributions[distribution[1]])
    biotechnicalProbability = sample_value(-11, -3, distributions[distribution[2]])
    return N - (astrophysicsProbability + biotechnicalProbability)


def get_point_model_3(max_n=10, distribution=(0, 0, 0, 0, 0, 0)):  # add expanding in universe to model 1
    # sample as in model 1
    RStarSample = sample_value(0, 2, distributions[distribution[0]])
    fPlanets = sample_value(-1, 0, distributions[distribution[1]])
    nEnvironment = sample_value(-1, 0, distributions[distribution[2]])
    fIntelligence = sample_value(-3, 0, distributions[distribution[3]])
    fCivilization = sample_value(-2, 0, distributions[distribution[4]])

    N = 10 ** sample_value(0, log10(max_n), distributions[distribution[5]])

    fLife = life_dist(mean=0, sigma=50)
    fLifeEks = float(log10(fLife))

    f = 10 ** (RStarSample + fPlanets + nEnvironment + fLifeEks + fIntelligence + fCivilization)

    # logL = log10(N) - log10(f)   ... model 1 would return logL like this

    A = 1
    B = 0.004 / (9.461e12 ** 3)  # number density of stars as per Wikipedia
    a4 = 5.13342 * 1e10 * 10 ** (fPlanets + nEnvironment) * B    # estimated number of earth-like planets
    a14 = f * A   # rate of new intelligent civilisation born
    candidates = list(roots([a4 * a14, 0, 0, a14, -N]))   # zeros of function: a4 * a14 * x^4 + a14 * x - N
    # actually we want to solve equation: f*A * (L + 5.13342*1e10*10**(fPlanets+nEnvironment)*B * L**4) = N
    L_initial_guess = N / (a14*log10(a14)**4)  # just a bad approximation to detect true candidate
    candidates.sort(key=lambda x: nabs(x - L_initial_guess))
    L = log10(candidates[0])
    return L


def get_point(max_n=10, distribution=(0, 0, 0, 0, 0, 0), model=1):   # get point for selected parameters
    if model == 1:
        return float(get_point_model_1(max_n, distribution).real)
    if model == 2:
        return float(get_point_model_2(max_n, distribution).real)
    if model == 3:
        return float(get_point_model_3(max_n, distribution).real)
