# generate points distributed by model to draw distribution of L(N)
from config import *
import numba as nb


# get log(L) for max_n at random other parameters
# ============ Models ====================

# model 1
def get_point_model_1(N=nrange, distribution=(0, 1, 0, 0, 0, 0), size=noIterations):
    # sample parameters in logarithmic scale
    RStarSample = sample_value(np.log10(2), 0, np.log10(5), distributions[distribution[0]], size)  # rate of new star born
    fPlanets = sample_value(np.log10(0.9), -2, 0, distributions[distribution[1]], size)  # probability that star has planets
    nEnvironment = sample_value(np.log10(2), 0, np.log10(5), distributions[distribution[2]], size)  # number of potentialy eart-like planets per star
    fIntelligence = sample_value(np.log10(0.9), np.log10(0.2), 0, distributions[distribution[3]], size)  # prob. some intelligent beings start to exist
    fCivilization = sample_value(-1, -2, 0, distributions[distribution[4]], size)  # prob. this beings are possible to communicate
    fLifeEks = sample_value(np.log10(0.9), np.log10(0.2), 0, distributions[distribution[5]], size)  # probability that life begins

    #       with other planets
    # N = RStarSample + fPlanets + nEnvironment + fLifeEks + fInteligence + fCivilization + L
    paramDist = RStarSample + fPlanets + nEnvironment + fLifeEks + fIntelligence + fCivilization

    if size == 1:
        return ["Drake", 0, RStarSample[0], fPlanets[0], 0, nEnvironment[0], fLifeEks[0],
                fIntelligence[0], fCivilization[0], 0, 0, 0, 0, 0, 0,
                np.log10(N)], (np.log10(N) - paramDist)[0]

    logL1 = [np.log10(n) - paramDist for n in N]
    return [logL1]


@nb.jit(nopython=True)
def solveL3(f, a4, N):
    l3 = []
    for n in N:
        aux = []
        for a, b in zip(f, a4):
            aux2 = 1e8
            for c in np.roots(np.array([b, 0, a, -n], dtype=np.complex128)):
                if np.abs(c.imag) < 1e-8 and np.abs(c.real) < aux2:
                    aux2 = np.abs(c.real)
            a1 = np.log10(aux2 + 1e-16)
            aux.append(a1)
        l3.append(aux)
    return np.array(l3).astype(np.float64)


# model 3, adds expanding in universe to model 1
def get_point_model_3(N=nrange, distribution=(0, 1, 0, 0, 0), size=noIterations):
    # sample parameters in logarithmic scale
    RStarSample = sample_value(np.log10(2), 0, np.log10(5), distributions[distribution[0]], size)  # rate of new star born
    fPlanets = sample_value(np.log10(0.9), -2, 0, distributions[distribution[1]], size)  # probability that star has planets
    nEnvironment = sample_value(np.log10(2), 0, np.log10(5), distributions[distribution[2]], size)  # number of potentialy eart-like planets per star
    fIntelligence = sample_value(np.log10(0.9), np.log10(0.2), 0, distributions[distribution[3]], size)  # prob. some intelligent beings start to exist
    fCivilization = sample_value(-1, -2, 0, distributions[distribution[4]], size)  # prob. this beings are possible to communicate
    fLifeEks = sample_value(np.log10(0.9), np.log10(0.2), 0, distributions[distribution[5]], size)  # probability that life begins

    #       with other planets
    # N = RStarSample + fPlanets + nEnvironment + fLifeEks + fInteligence + fCivilization + L
    paramDist = RStarSample + fPlanets + nEnvironment + fLifeEks + fIntelligence + fCivilization

    f = 10 ** paramDist
    # logL = log10(N) - log10(f)   ... model 1 would return logL like this
    B = 0.004  # number density of stars per cubic light year from Wikipedia
    a4 = 10 ** (fPlanets + nEnvironment) * B * np.pi  # estimated number of earth-like planets per pi square light years
    a4 = 1e-6 * a4 * f  # we are moving with 0.1 % of light speed
    # model 3:
    # zeros of: f * (1e-6 * a4 * x^3 + x) - N
    # actually we want to solve equation: f*L*(1 + A * pi * 1e-6*10**(fPlanets+nEnvironment)*B * L**2) = N
    logL3 = solveL3(f, a4, np.array(N))

    if size == 1:
        return ["Expand", 0, RStarSample[0], fPlanets[0], 0, nEnvironment[0], fLifeEks[0],
                fIntelligence[0], fCivilization[0], 0, 0, 0, 0, 0, 0,
                np.log10(N[0])], logL3[0][0]
    return [logL3]


def get_point_model_2(N=nrange, distribution=(0, 0), size=noIterations):
    astrophysicsProbability = sample_value(np.log10(3.6), -2, np.log10(25), distributions[distribution[0]], size)
    biotechnicalProbability = sample_value(np.log10(0.081), -4.5, 0, distributions[distribution[1]], size)
    if size == 1:
        return ["Simplified", astrophysicsProbability[0], 0, 0, biotechnicalProbability[0],
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                np.log10(N)], (np.log10(N) - (astrophysicsProbability + biotechnicalProbability)).real[0]
    return [[(np.log10(n) - (astrophysicsProbability + biotechnicalProbability)).real for n in N]]


# rare earth theory
def get_point_model_4(N=nrange, distribution=(0, 0, 0, 0, 0, 0, 0, 0), size=noIterations):
    RStarSample = sample_value(np.log10(2), 0, np.log10(5), distributions[distribution[0]],
                               size)  # rate of new star born
    nEnvironment = sample_value(np.log10(2), 0, np.log10(5), distributions[distribution[1]],
                                size)  # number of potentialy eart-like planets per star

    # f_l * f_i * f_c from Drake is equal to f_i * f_c * f_l from RareEarth equation
    DrakesParam = RStarSample + nEnvironment

    # rare earth equation
    Nzvezdica_ne = sample_value(np.log10(5e11), np.log10(5e10), np.log10(5e12), distributions[distribution[2]], size)
    fGalactic = sample_value(-1, np.log10(0.05), np.log10(0.15), distributions[distribution[3]], size)
    fMetal = sample_value(-1, -2, np.log10(0.2), distributions[distribution[4]], size)

    fMoon = sample_value(-2, -2.5, -1.5, distributions[distribution[5]], size)
    fJupiter = sample_value(np.log10(0.8), -1, 0, distributions[distribution[6]], size)
    fNoExtinction = sample_value(-2, -2.5, -1.5, distributions[distribution[7]], size)

    L = (Nzvezdica_ne + fGalactic + fMetal + fMoon + fJupiter + fNoExtinction) - DrakesParam
    if size == 1:
        return ["Rare_Earth", 0, RStarSample[0], 0, 0, nEnvironment[0], 0, 0, 0,
                Nzvezdica_ne[0], fGalactic[0], fMetal[0], fMoon[0], fJupiter[0], fNoExtinction[0],
                np.log10(N)], L[0].real
    return [[L.real for _ in N]]


def get_point(Ns=nrange, distribution=(0, 0, 0, 0, 0), model=(1, 3)):  # get point for selected parameters
    rezultati = []
    if 1 in model:
        rezultati += get_point_model_1(Ns, distribution)
    if 2 in model:
        rezultati += get_point_model_2(Ns, distribution)
    if 3 in model:
        rezultati += get_point_model_3(Ns, distribution)
    if 4 in model:
        rezultati += get_point_model_4(Ns, distribution)
    rHist = [[np.histogram(r, bin_no, (-1, maxLogL))[0] for r in rez]
             for rez in rezultati]
    hists = [[] for _ in model]  # make some initially empty lists
    for m in range(len(model)):
        for n in range(len(Ns)):
            hists[m].append(rHist[m][n] / noIterations)
    return np.array(hists)


def random_point(supermodel_1=True):
    j = np.random.randint(0, 4)
    n1 = 10 ** sample_value(0.5, 0, 3, distributions[np.random.randint(0, len(distributions), 1)[0]], 1)[0]
    n2 = 10 ** sample_value(6, 0, 8, ["loglinear", "uniform", "loguniform"][np.random.randint(0, 3, 1)[0]], 1)[0]
    n = n1 if supermodel_1 else n2
    if j == 0:
        return get_point_model_1(n1, np.random.randint(0, len(distributions), 6), 1)
    elif j == 1:
        return get_point_model_2(n1, np.random.randint(0, len(distributions), 2), 1)
    elif j == 2:
        return get_point_model_3([n], np.random.randint(0, len(distributions), 6), 1)
    else:
        return get_point_model_4(n1, np.random.randint(0, len(distributions), 8), 1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ns = [1, 1000, 1000000]
    plt.figure(figsize=(12, 12), tight_layout=True)
    for d in range(3):
        distribution = tuple([d] * 8)
        pt = get_point(ns, distribution, (1, 2, 3, 4))
        for i, n in enumerate(ns):
            plt.subplot(len(ns), 3, d + i * 3 + 1)
            plt.title(f"{distributions[d]}, N = {n}")
            for j in [1, 2, 3, 4]:
                if j in [1, 3]:
                    plt.plot(xLogL, pt[j-1][i, :].T, label=j)
            plt.grid()
            plt.legend()
    plt.show()
    print("Done!")
