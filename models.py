# generate points distributed by model to draw distribution of L(N)
from numpy.random import uniform, normal, lognormal
from numpy import log10, roots, abs as nabs, exp as nexp

# all possible distributions and models (lists for iterating)
distributions = ["loguniform", "uniform", "halfgauss", "lognormal", "fixed"]  # all possible distributions
models = [1, 2, 3, 4]


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

# model 1 and 3, model 3 adds expanding in universe to model 1
def get_point_model_1_3(model=(1, 3), max_n=10, distribution=(0, 0, 0, 0, 0, 0)):
    # sample parameters in logarithmic scale
    RStarSample = sample_value(0, 2, distributions[distribution[0]])  # rate of new star born
    fPlanets = sample_value(-1, 0, distributions[distribution[1]])  # probability that star has planets
    nEnvironment = sample_value(-1, 0, distributions[distribution[2]])  # probability that it is earth-like
    fIntelligence = sample_value(-3, 0, distributions[distribution[3]])  # prob. some intelligent beings start to exist
    fCivilization = sample_value(-2, 0, distributions[distribution[4]])  # prob. this beings are possible to communicate
    #       with other planets

    logN = sample_value(0, log10(max_n), distributions[distribution[5]])
    fLife = life_dist(mean=0, sigma=50)  # probability that life begins
    fLifeEks = log10(fLife)

    # N = RStarSample + fPlanets + nEnvironment + fLifeEks + fInteligence + fCivilization + L
    logL = logN - (RStarSample + fPlanets + nEnvironment + fLifeEks + fIntelligence + fCivilization)
    if 3 not in model:  # if only model 1
        return [float(logL.real)]  # rate of birth of new civilisation

    N = 10 ** logN
    f = 10 ** (RStarSample + fPlanets + nEnvironment + fLifeEks + fIntelligence + fCivilization)
    # logL = log10(N) - log10(f)   ... model 1 would return logL like this
    A = 1
    B = 0.004 / (9.461e12 ** 3)  # number density of stars as per Wikipedia
    a4 = 5.13342 * 1e10 * 10 ** (fPlanets + nEnvironment) * B  # estimated number of earth-like planets
    a14 = f * A  # rate of new intelligent civilisation born
    candidates = list(roots([a4 * a14, 0, 0, a14, -N]))  # zeros of function: a4 * a14 * x^4 + a14 * x - N
    # actually we want to solve equation: f*A * (L + 5.13342*1e10*10**(fPlanets+nEnvironment)*B * L**4) = N
    L_initial_guess = N / (a14 * log10(a14) ** 4)  # just a bad approximation to detect true candidate
    candidates.sort(key=lambda x: nabs(x - L_initial_guess))
    logL3 = log10(candidates[0])
    if 1 in model:
        return [float(logL.real), float(logL3.real)]
    return [float(logL3.real)]


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
    return [float(N - (astrophysicsProbability + biotechnicalProbability)).real]


# rare earth theory
def get_point_model_4(distribution=(0, 0, 0, 0, 0, 0, 0)):
    RStarSample = sample_value(0, 2, distributions[distribution[0]])
    nStars = uniform(11, 11.60205999132)
    fMetal = sample_value(-2, -1, distributions[distribution[1]])
    ngalactic = sample_value(10.3, 10.6, distributions[distribution[2]])  # about 20 to 40 billion stars.
    fMoon = sample_value(-2.5, -1.5, distributions[distribution[3]])
    fExtinct = sample_value(-2, -1, distributions[distribution[4]])
    fj = sample_value(-1.3, -1, distributions[distribution[5]])
    Nzvezdica_ne = sample_value(11.5, 11.9, distributions[distribution[6]])
    # Drake equation using Rare Earth equation
    L = (Nzvezdica_ne + fMetal + ngalactic + fMoon + fExtinct + fj) - (RStarSample + nStars)
    return [float(L).real]


def get_point(max_n=10, distribution=(0, 0, 0, 0, 0, 0), model=(1, 3)):  # get point for selected parameters
    if 2 in model:
        return get_point_model_2(max_n, distribution)
    if 1 in model or 3 in model:
        return get_point_model_1_3(model, max_n, distribution)
    if 4 in model:
        return get_point_model_4(distribution)
