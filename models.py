from config import *
from scipy import interpolate


# draw comparison of Model I and Model III
def L1toL3():
    xi = np.linspace(1, 10, 100)  # for different L1 and values of f_p + n_e
    minP = bounds["f_p"][1] + bounds["n_e"][1]
    meanP = bounds["f_p"][0] + bounds["n_e"][0]
    maxP = bounds["f_p"][2] + bounds["n_e"][2]
    a1 = 1e-6 * 10 ** np.array([minP, meanP, maxP]) * np.pi * 0.004
    h = np.log10(np.abs([[- min(np.roots([a, 0, 1, -L1]), key=lambda x: np.abs(x.imag)).real
                          for a in a1] for L1 in 10 ** xi]))
    ca = [("tab:orange", 0.3), ("tab:red", 1), ("tab:orange", 0.3)]
    for ylab, ydat in [("L_3", h)]:
        plt.figure(figsize=(6, 4), dpi=200, tight_layout=True)
        plt.fill_between(xi, ydat[:, 0], ydat[:, 2],
                         color=ca[0][0], alpha=ca[0][1] / 2)
        for i in range(3):
            plt.plot(xi, ydat[:, i], color=ca[i][0], alpha=ca[i][1])
        if "L_1" not in ylab:
            plt.plot(xi, xi, "--", alpha=0.7)
        plt.title(f"${ylab}(L_1)$")
        plt.xlabel("$L_1$")
        plt.ylabel(f"${ylab}$")
        plt.grid()
        plt.savefig(f"out/model3-model1.png")
        plt.show()


# interpolate value of L3 from value of L1 and f_p + n_e
interp_logL1 = np.linspace(-3, 12, 15 * 128)
interp_inva4 = np.linspace(bounds["f_p"][1] + bounds["n_e"][1] - 0.1,
                           bounds["f_p"][2] + bounds["n_e"][2] + 0.1, 64)
interp_file = f"{collected_folder}/arrayL3.npy"   # write values in file
if os.path.exists(interp_file):
    array = np.load(interp_file)
else:
    array = np.log10(np.abs(np.array([[- min(np.roots([a4, 0, 1, -l1]),   # solve equation for L3
                                             key=lambda x: np.abs(x.imag)).real for l1 in 10 ** interp_logL1]
                                      for a4 in 1e-6 * 0.004 * np.pi * 10 ** interp_inva4])))
    np.save(interp_file, array)
interpolator = interpolate.interp2d(interp_logL1, interp_inva4, array, kind='linear')


def findElementsL3(distL1, inva4, N):
    return np.array([interpolator(N - l1, a4) for a4, l1 in zip(inva4, distL1)]).T


# Model I
# Model II, simplified Model I
# Model III, adds expanding in universe to Model I
# Model IV, rare Earth theory
def get_point_models(N=nrange2, distribution=(0, 0, 0, 0, 0, 0, 0, 0), size=noIterations):
    # Model I and III parameters
    R_star = sample_value(bounds["R_*"], distributions[distribution[0]], size)  # rate of new star born
    f_p = sample_value(bounds["f_p"], distributions[distribution[1]], size)  # probability that star has planets
    n_e = sample_value(bounds["n_e"], distributions[distribution[2]], size)  # number of potentialy eart-like planets per star
    f_l = sample_value(bounds["f_l"], distributions[distribution[3]], size)  # probability that life begins
    f_i = sample_value(bounds["f_i"], distributions[distribution[4]], size)  # prob. some intelligent beings start to exist
    f_c = sample_value(bounds["f_c"], distributions[distribution[5]], size)  # prob. this beings are possible to communicate
    # Model II parameters
    f_a = sample_value(bounds["f_a"], distributions[distribution[1]], size)
    f_b = sample_value(bounds["f_b"], distributions[distribution[5]], size)
    # Model IV parameters (f_l * f_i * f_c from Model I is equal to f_i * f_c * f_l from Model IV)
    Nzvezdica_ne = sample_value(bounds["N_* n_e"], distributions[distribution[1]], size)
    f_g = sample_value(bounds["f_g"], distributions[distribution[3]], size)
    f_pm = sample_value(bounds["f_{pm}"], distributions[distribution[4]], size)
    f_m = sample_value(bounds["f_m"], distributions[distribution[5]], size)
    f_j = sample_value(bounds["f_j"], distributions[int(np.random.randint(0, 3, 1)[0])], size)
    f_me = sample_value(bounds["f_{me}"], distributions[int(np.random.randint(0, 3, 1)[0])], size)

    # Model I and III partial sums
    phisic = R_star + f_p + n_e
    bio = f_l + f_i + f_c
    # Model IV rare Earth equation
    logL4 = (Nzvezdica_ne + f_g + f_pm + f_m + f_j + f_me) - (R_star + n_e)

    # f = 10 ** (phisic + bio)
    # B = 0.004  # number density of stars per cubic light year from Wikipedia
    # a4 = 10 ** (f_p + n_e) * B * np.pi  # estimated number of earth-like planets per pi square light years
    # a4 = 1e-6 * a4  # we are moving with 0.1 % of light speed in plane for square of speed (per square light year)
    # model 3:
    # find real and negative zero of function: a4 * x^3 + x - N / f
    # actually we want to solve equation: f * L * (1 + pi * 1e-6 * 10 ** (f_p + n_e) * B * L ** 2) = N
    if size == 1:
        # estimated missing irrelevant parameters
        auxfa = (f_a[0] - bounds["f_a"][1]) / (bounds["f_a"][2] - bounds["f_a"][1])
        auxfb = (f_b[0] - bounds["f_b"][1]) / (bounds["f_b"][2] - bounds["f_b"][1])
        auxphisic = [(bounds[ph][2] - bounds[ph][1]) * auxfa + bounds[ph][1] for ph in ["R_*", "f_p", "n_e"]]
        auxbio = [(bounds[ph][2] - bounds[ph][1]) * auxfb + bounds[ph][1] for ph in ["f_l", "f_i", "f_c"]]
        return [[([sm, 1, 0, 0, 0, np.log10(N[0]),   # attributes for ML
                  phisic[0], R_star[0], f_p[0], n_e[0],
                  bio[0], f_l[0], f_i[0], f_c[0],
                  Nzvezdica_ne[0], f_g[0], f_pm[0], f_m[0], f_j[0], f_me[0]], np.log10(N[0]) - (phisic + bio)[0]),
                ([sm, 0, 1, 0, 0, np.log10(N[0]),
                  f_a[0], auxphisic[0], auxphisic[1], auxphisic[2],
                  f_b[0], auxbio[0], auxbio[1], auxbio[2],
                  Nzvezdica_ne[0], f_g[0], f_pm[0], f_m[0], f_j[0], f_me[0]], np.log10(N[0]) - (f_a + f_b)[0]),
                ([sm, 0, 0, 1, 0, np.log10(N[sm - 1]),
                  phisic[0], R_star[0], f_p[0], n_e[0],
                  bio[0], f_l[0], f_i[0], f_c[0],
                  Nzvezdica_ne[0], f_g[0], f_pm[0], f_m[0], f_j[0], f_me[0]],
                 findElementsL3([(phisic + bio)[:1]], (f_p + n_e)[:1], np.log10(N[sm - 1]))[0][0]),
                ([sm, 0, 0, 0, 1, np.log10(N[0]),
                  phisic[0], R_star[0], f_p[0], n_e[0],
                  bio[0], f_l[0], f_i[0], f_c[0],
                  Nzvezdica_ne[0], f_g[0], f_pm[0], f_m[0], f_j[0], f_me[0]], logL4[0])] for sm in [1, 2]]
    # make histograms
    logL1 = [np.histogram(np.log10(n) - (phisic + bio), bin_no, (-1, maxLogL))[0] / noIterations for n in N]
    logL2 = [np.histogram(np.log10(n) - (f_a + f_b), bin_no, (-1, maxLogL))[0] / noIterations for n in N]
    logL3 = [np.histogram(r, bin_no, (-1, maxLogL))[0] / noIterations
             for r in findElementsL3(phisic + bio, f_p + n_e, np.log10(N))]
    logL4 = [np.histogram(logL4, bin_no, (-1, maxLogL))[0] / noIterations] * len(N)
    return [logL1, logL2, logL3, logL4]


def get_point(Ns=nrange2, distribution=(0, 0, 0, 0, 0, 0, 0, 0)):  # get histograms for selected distribution
    return np.array(get_point_models(Ns, distribution))


def random_point():   # get only random shots for ML and not whole histograms
    n1 = 10 ** sample_value(bounds["N"], distributions[np.random.randint(0, len(distributions), 1)[0]], 1)[0]
    n2 = 10 ** sample_value(bounds["N2"], ["loglinear", "lognormal2", "loguniform"][np.random.randint(0, 3, 1)[0]], 1)[0]
    points = get_point_models([n1, n2], np.random.randint(0, len(distributions), 6), 1)
    return points[0] + points[1]
