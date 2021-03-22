
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from scipy.optimize import curve_fit

from kepler import fetch_single
from timer import EasyTimer as Timer


t_star = 0.22503650524036528
r_bar = 2.2085976600646973


def make_cumulative(x):
    return np.concatenate(([0], np.cumsum(x)))


def __read_file(filepath, keys):
    with open(filepath, "r") as f:
        header = f.readline()    

        df = pd.read_csv(f, sep=r"\s+", names=keys)

    return dict(df)


def read_binaries(path, time):
    file = path / Path("bev.82_" + str(time))
    keys = ["time", "j1", "j2", "name1", "name2", "kw1", "kw2", "kcm",
            "ri", "ecc", "pb", "semi", "m1", "m2", "zl1", "zl2", "r1",
            "r2", "te1", "te2"]

    return __read_file(file, keys)


def read_singles(path, time):
    file = path / Path("sev.83_" + str(time))
    keys = ["time", "i", "name", "kw", "ri", "m", "zl", "r", "te"]

    try:
        return __read_file(file, keys)
    except FileNotFoundError:
        fetch_single(file)

    return __read_file(file, keys)


def mask_dict(d, m):
    keys = d.keys()
    values = (a[m] for a in d.values())

    result = dict(zip(keys, values))

    return result


def clean_binaries(data):
    valid = (data["te1"] < 4.9) \
            & (data["te2"] < 4.9) \
            & (np.abs(data["m1"]) > 1e-10) \
            & (np.abs(data["m2"]) > 1e-10)

    print(f"Masking {len(valid) - np.sum(valid)} Invalid Binaries.")

    # incorrect r_i in binaries has to be corrected
    data["ri"] = r_bar ** 2 * data["ri"]

    return mask_dict(data, valid)


def clean_singles(data):
    valid = (data["te"] < 4.9) & (np.abs(data["m"]) > 1e-10)
    print(f"Masking {len(valid) - np.sum(valid)} Invalid Singles.")

    return mask_dict(data, valid)


def init_data(path, time, which="all", clean=True):
    if which == "all":
        bdata = read_binaries(path, time)
        sdata = read_singles(path, time)
    elif which == "binaries":
        bdata = read_binaries(path, time)
        sdata = None
    elif which == "singles":
        bdata = None
        sdata = read_singles(path, time)
    else:
        raise ValueError(f"Invalid 'which': {which}")
    if clean:
        if sdata:
            sdata = clean_singles(sdata)
        if bdata:
            bdata = clean_binaries(bdata)

    assert (bdata is not None) or (sdata is not None)

    return bdata, sdata


def tlm(path, time, which="all", clean=True):
    bdata, sdata = init_data(path, time, which, clean)

    if bdata and sdata:
        t = np.concatenate((bdata["te1"], bdata["te2"], sdata["te"]))
        l = np.concatenate((bdata["zl1"], bdata["zl2"], sdata["zl"]))
        m = np.concatenate((bdata["m1"], bdata["m2"], sdata["m"]))
    elif bdata:
        t = np.concatenate((bdata["te1"], bdata["te2"]))
        l = np.concatenate((bdata["zl1"], bdata["zl2"]))
        m = np.concatenate((bdata["m1"], bdata["m2"]))
    else:
        t = sdata["te"]
        l = sdata["zl"]
        m = sdata["m"]

    return t, l, m


def save_plot(clean, which, time, name, fmt="png"):
    if time is None:
        time = ""
    path = Path("plots_new") / Path(f"{'clean' if clean else 'raw'}{which}{time}{name}.{fmt}")
    print(f"Saving plot '{path}'...")
    plt.savefig(path)



def hrd(path, time, which="all", clean=True, save=True):
    t, l, m = tlm(path, time, which, clean)

    plt.scatter(t, l, marker=".", c=np.log10(m))

    plt.xlim(reversed(plt.xlim()))
    plt.title(f"HRD at $t = {time * t_star * 1e-3:.1f}$ Gyr")
    plt.xlabel(r"$\log_{10} T_{eff}$")
    plt.ylabel(r"$\log_{10} L / L_{\odot}$")
    plt.colorbar(label=r"$\log_{10} M / M_{\odot}$")

    plt.tight_layout()

    if save:
        save_plot(clean, which, time, "hrd")


def binary_fraction(path, times, clean=True, save=True):
    fractions = []

    for time in times:
        bdata, sdata = init_data(path, time, which="all", clean=clean)

        nb = len(bdata["time"])
        ns = len(sdata["time"])
        fractions.append(nb / (nb + ns))

    fractions = np.array(fractions)

    times = 1e-3 * t_star * times

    # use last 30% for fit
    n = int(0.3 * len(fractions))
    linear_x = times[-n:]
    linear_y = fractions[-n:]

    def fit_func(x, a, b):
        return a * x + b

    popt, pcov = curve_fit(fit_func, linear_x, linear_y)

    plt.plot(times, fractions, label="Binary Fraction")
    plt.plot(times, fit_func(times, *popt), lw=1, label="Linear Asymptote")
    plt.xlabel(f"Time / Gyr")
    plt.ylabel(f"Binary Fraction")

    plt.title("Binary Fraction over Time")

    plt.legend()

    plt.tight_layout()

    if save:
        save_plot(clean, "all", None, "bf")


def radial_distributions(path, time, which="all", clean=True, save=True):
    bdata, sdata = init_data(path, time, which=which, clean=clean)

    if bdata and sdata:
        plt.hist([sdata["ri"], bdata["ri"]], bins=100, stacked=True)
        plt.legend(["Singles", "Binaries"])
    elif bdata:
        plt.hist(bdata["ri"], bins=100)
    else:
        plt.hist(sdata["ri"], bins=100)    

    plt.tight_layout()

    if save:
        save_plot(clean, which, time, "rdist")


# def radial_cfractions(path, time, clean=True, save=True):
#     bdata, sdata = init_data(path, time, which="all", clean=clean)

#     nb = len(bdata["time"])
#     ns = len(sdata["time"])

#     rs = np.concatenate((bdata["ri"], sdata["ri"]))

#     n_bins = 100
#     bins = np.linspace(0, np.max(rs), n_bins)

#     bcounts, _ = np.histogram(bdata["ri"], bins)
#     scounts, _ = np.histogram(sdata["ri"], bins)

#     cum_binaries = np.cumsum(bcounts)
#     cum_singles = np.cumsum(scounts)

#     bfractions = cum_binaries / (ns + cum_binaries)
#     sfractions = cum_singles / (cum_singles + nb)

#     bfractions = np.concatenate(([0], bfractions))
#     sfractions = np.concatenate(([0], sfractions))

#     fig = plt.gcf()
#     ax1 = fig.add_subplot(1, 1, 1)
#     bline, = ax1.plot(bins, bfractions, label="Cumulative Binaries Fraction", color="C0")

#     ax2 = ax1.twinx()
#     sline, = ax2.plot(bins, sfractions, label="Cumulative Singles Fraction", color="C1")

#     ax1.set_xlabel("Distance from Center")
#     ax1.set_ylabel("Binaries Fraction")
#     ax2.set_ylabel("Singles Fraction")
#     plt.legend((bline, sline), ["Binaries", "Singles"])

#     fig.tight_layout()

#     if save:
#         save_plot(clean, "all", time, "rbf")


def radial_hist(path, time, clean=True, save=True):
    bdata, sdata = init_data(path, time, which="all", clean=clean)

    smean = np.mean(sdata["ri"])
    bmean = np.mean(bdata["ri"])
    amean = np.mean(np.concatenate((sdata["ri"], bdata["ri"])))

    fig = plt.gcf()
    ax1 = fig.add_subplot(1, 1, 1)

    plt.axvline(x=amean, alpha=0.3, color="black")
    plt.axvline(x=smean, alpha=0.3, color="C0")
    plt.axvline(x=bmean, alpha=0.3, color="C1")

    r_max = 75
    smask = sdata["ri"] <= r_max
    bmask = bdata["ri"] <= r_max


    ax1.hist((sdata["ri"][smask], bdata["ri"][bmask]), bins=500, stacked=True)
    ax1.set_xlabel("Distance from Center in pc")
    ax1.set_ylabel("Number of Stars")

    plt.title(f"Stacked Histogram at $t = {1e-3 * t_star * time:.1f}$Gyr")
    plt.legend([rf"$\mu_{{all}} = {amean:.2f}$pc", rf"$\mu_S = {smean:.2f}$pc", rf"$\mu_B = {bmean:.2f}$pc", "Singles", "Binaries"])

    plt.xlim(0, 75)

    fig.tight_layout()

    if save:
        save_plot(clean, "all", time, "rhist")


def mass_dist(path, time, clean=True, save=True):
    bdata, sdata = init_data(path, time, which="all", clean=clean)

    masses = np.concatenate((bdata["m1"] + bdata["m2"], sdata["m"]))
    radii = np.concatenate((bdata["ri"], sdata["ri"]))

    sort = np.argsort(radii)
    radii = radii[sort]
    masses = masses[sort]

    radii = np.concatenate(([0], radii))
    cmass = make_cumulative(masses)

    # half mass radius
    mass_fractions = cmass / cmass[-1]
    hmr_idx = np.argmax(mass_fractions >= 0.5)
    hmr = radii[hmr_idx]

    plt.plot(radii, cmass)
    plt.xlabel("Distance from Center / pc")
    plt.ylabel(r"Cumulative Mass $M / M_{\odot}$")
    plt.title(f"Cumulative Radial Mass Distribution at $t = {1e-3 * t_star * time:.1f}$Gyr")

    plt.axvline(hmr, alpha=0.2, color="black", label=f"$R_{{1/2}} = {hmr:.2f}$pc")

    plt.legend()

    if save:
        save_plot(clean, "all", time, "mdist")


def mass_seg(path, time, which="all", clean=True, save=True):
    # mass segregation: stars in the center are more massive than stars outside
    bdata, sdata = init_data(path, time, which="all", clean=clean)

    if which == "all":
        masses = np.concatenate((bdata["m1"] + bdata["m2"], sdata["m"]))
        radii = np.concatenate((bdata["ri"], sdata["ri"]))
    elif which == "singles":
        masses = sdata["m"]
        radii = sdata["ri"]
    else:
        masses = bdata["m1"] + bdata["m2"]
        radii = bdata["ri"]

    if which == "singles" or which == "all":
        plt.scatter(sdata["m"], sdata["ri"], marker=".", s=1, label="Singles", color="C0")

    if which == "binaries" or which == "all":
        plt.scatter(bdata["m1"] + bdata["m2"], bdata["ri"], marker=".", s=1, label="Binaries", color="C1")

    def fit_func(x, a, b):
        return a * x ** b

    x = np.logspace(np.log10(np.min(masses)), np.log10(np.max(masses)), 1000, base=10)

    if which == "singles" or which == "all":
        popt, pcov = curve_fit(fit_func, sdata["m"], sdata["ri"])
        plt.plot(x, fit_func(x, *popt), label="Singles Trend", color="C0")

    if which == "binaries" or which == "all":
        popt, pcov = curve_fit(fit_func, bdata["m1"] + bdata["m2"], bdata["ri"])
        plt.plot(x, fit_func(x, *popt), label="Binaries Trend", color="C1")

    if which == "all":
        popt, pcov = curve_fit(fit_func, masses, radii)
        plt.plot(x, fit_func(x, *popt), label="Overall Trend", color="C2")

    plt.xlabel(r"$M / M_{\odot}$")
    plt.ylabel(r"Distance from Center / pc")

    if time < 10000:
        t = f"$t={t_star * time:.1f}$Myr"
    else:
        t = f"$t={1e-3 * t_star * time:.1f}$Gyr"

    plt.title(f"Mass Segregation at {t}")
    plt.legend(loc="upper right")

    plt.xscale("log")
    plt.yscale("log")

    plt.tight_layout()

    if save:
        save_plot(clean, "all", time, "mseg")




def mark_half_maximum(bins, counts, cumulative=True, horizontal=True, vertical=True, hlabel=None, vlabel=None):
    if not cumulative:
        counts = make_cumulative(counts)
    fractions = counts / counts[-1]
    hm = np.argmax(fractions >= 0.5)

    # fixate x and y limits
    limits = plt.axis()

    if horizontal:
        plt.hlines(0.5, xmin=0, xmax=bins[hm], color="black", alpha=0.2, label=hlabel)

    if vertical:
        plt.vlines(bins[hm], ymin=-1, ymax=0.5, color="black", alpha=0.2, label=vlabel)


    # fixate x and y limits
    plt.axis(limits)




def radial_cfractions(path, time, clean=True, save=True):
    bdata, sdata = init_data(path, time, which="all", clean=clean)

    nb = len(bdata["time"])
    ns = len(sdata["time"])

    rs = np.concatenate((bdata["ri"], sdata["ri"]))

    n_bins = 500
    bins = np.linspace(0, np.max(rs), n_bins)

    # having this constant gives all times the same x limits
    bins = np.logspace(-2, 3, n_bins, base=10)
    # bins = np.logspace(np.log10(np.min(rs)), np.log10(np.max(rs)), n_bins, base=10)
    
    bcounts, _ = np.histogram(bdata["ri"], bins)
    scounts, _ = np.histogram(sdata["ri"], bins)

    cum_binaries = make_cumulative(bcounts)
    cum_singles = make_cumulative(scounts)

    bfractions = cum_binaries / nb
    sfractions = cum_singles / ns

    # half maxima
    bhm = np.argmax(bfractions >= 0.5)
    shm = np.argmax(sfractions >= 0.5)

    plt.plot(bins, sfractions, label="Singles", color="C0")
    plt.plot(bins, bfractions, label="Binaries", color="C1")

    # have to set scale to logarithmic before getting limits
    plt.xscale("log")

    mark_half_maximum(bins, cum_binaries, horizontal=False, vlabel=f"$R_{{B 1/2}} = {bins[bhm]:.2f}$pc")
    mark_half_maximum(bins, cum_singles, horizontal=True, vlabel=f"$R_{{S 1/2}} = {bins[shm]:.2f}$pc")

    plt.xlabel("Distance from Center in pc")
    plt.ylabel("Fraction")

    plt.title(f"Cumulative Fractions at $t={1e-3 * t_star * time:.1f}$Gyr")

    plt.legend()
    plt.tight_layout()

    if save:
        save_plot(clean, "all", time, "rcf")



def radial_remnants(path, time, clean=True, save=True):
    bdata, _ = init_data(path, time, which="binaries", clean=clean)

    # remnants have stellar type 10-14
    remnants = ((bdata["kw1"] >= 10) & (bdata["kw1"] <= 14)) | ((bdata["kw2"] >= 10) & (bdata["kw2"] <= 14))

    n_binaries = len(remnants)
    n_remnants = np.sum(remnants)

    if n_remnants == 0:
        return

    rdata = mask_dict(bdata, remnants)

    print(f"{np.sum(remnants)} out of {n_binaries} Binaries have remnants.")

    n_bins = 250
    bins = np.logspace(np.log10(np.min(bdata["ri"])), np.log10(np.max(bdata["ri"])), n_bins, base=10)

    radii, _ = np.histogram(bdata["ri"], bins)
    rradii, _ = np.histogram(rdata["ri"], bins)

    cum_radii = make_cumulative(radii)
    cum_rradii = make_cumulative(rradii)

    cum_fractions = cum_radii / n_binaries
    cum_rfractions = cum_rradii / n_remnants

    plt.plot(bins, cum_fractions, label="All Binaries")
    plt.plot(bins, cum_rfractions, label="with Remnant")
    plt.xlabel("Distance from Center in pc")
    plt.ylabel("Cumulative Fraction of Binaries")

    plt.title(f"Cumulative Radial Fractions\nat $t={1e-3 * t_star * time:.1f}$Gyr")

    plt.xscale("log")
    plt.legend()

    mark_half_maximum(bins, cum_rradii, horizontal=False)
    mark_half_maximum(bins, cum_radii, horizontal=True)

    plt.tight_layout()

    if save:
        save_plot(clean, "binaries", time, "remnants")


def _semi_ri(path, time, clean=True, n_bins=20):
    bdata, _ = init_data(path, time, which="binaries", clean=clean)

    x = np.log10(bdata["ri"])
    y = bdata["semi"]

    x_bins = np.linspace(np.min(x), np.max(x), n_bins)
    y_bins = np.linspace(np.min(y), np.max(y), n_bins)
    z, *_ = np.histogram2d(x, y, n_bins)

    return x_bins, y_bins, z


def semi_ri3d(path, time, clean=True, save=True):
    n_bins = 20
    x_bins, y_bins, z = _semi_ri(path, time, clean, n_bins)
    xx, yy = np.meshgrid(x_bins, y_bins)

    fig = plt.gcf()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ax.plot_surface(xx, yy, z, cmap="viridis")
    ax.set_xlabel(r"$\log_{10} R_i / 1$pc", labelpad=20)
    ax.set_ylabel(r"$\log_{10} a / R_{\odot}$", labelpad=20)
    ax.set_zlabel("counts", labelpad=20)

    ax.set_xlim(reversed(ax.get_xlim()))

    plt.title(f"Binary Distribution in Semi-Major Axis $a$\nvs. Distance from Center $R_i$\nat $t = {1e-3 * t_star * time:.1f}$Gyr")

    fig.tight_layout()

    if save:
        save_plot(clean, "binaries", time, "semi_ri3d")


def semi_ri2d(path, time, clean=True, save=True):
    n_bins = 100
    _, _, z = _semi_ri(path, time, clean, n_bins)

    plt.imshow(z)
    plt.xlabel(r"$\log_{10} R_i / 1$pc")
    plt.ylabel(r"$\log_{10} a / R_{\odot}$")

    plt.title(f"Binary Distribution in Semi-Major Axis $a$\nvs. Distance from Center $R_i$\nat $t = {1e-3 * t_star * time:.1f}$Gyr")

    plt.colorbar()

    plt.tight_layout()

    if save:
        save_plot(clean, "binaries", time, "semi_ri2d")


def lagrange_radii(path, times, fraction=0.5, clean=True, save=True):

    lrs = []

    for time in times:
        print(f"processing lagrange_radii at time {time}")
        bdata, sdata = init_data(path, time, which="all", clean=clean)

        ms = np.concatenate((bdata["m1"] + bdata["m2"], sdata["m"]))
        rs = np.concatenate((bdata["ri"], sdata["ri"]))

        sort = np.argsort(rs)
        rs = rs[sort]
        ms = ms[sort]

        cms = np.cumsum(ms)

        cms_fracs = cms / cms[-1]

        # lagrange radius: n% of mass
        lr = np.argmax(cms_fracs >= fraction)

        lrs.append(lr)

    plt.plot(1e-3 * t_star * times, lrs)
    plt.xlabel("Time / Gyr")
    plt.ylabel(f"{100 * fraction:.0f}% Lagrangian Radius / pc")

    plt.ylim(1)

    plt.yscale("log")
    
    plt.tight_layout()

    if save:
        save_plot(clean, "all", None, "rlag")


def stellar_types(path, time, which="all", clean=True, save=True):
    bdata, sdata = init_data(path, time, which=which, clean=clean)

    if which == "all":
        kw = np.concatenate((bdata["kw1"], bdata["kw2"], sdata["kw"]))
        twhich = "stars"
    elif which == "binaries":
        kw = np.concatenate((bdata["kw1"], bdata["kw2"]))
        twhich = which
    else:
        kw = sdata["kw"]
        twhich = which

    types = np.arange(0, 15 + 1)
    counts = []

    n = len(kw)

    for tp in types:
        counts.append(np.sum(kw == tp))

    counts = np.array(counts)

    plt.bar(types, counts)
    plt.xlabel("Stellar Type")
    plt.ylabel("Count")

    plt.title(f"Distribution of Stellar Types in {n} {twhich.capitalize()}\nat $t = {1e-3 * t_star * time:.1f}$Gyr")

    plt.tight_layout()

    if save:
        save_plot(clean, which, time, "kw")


def follow_star_hrd(path, times, name, which="singles", clean=True, save=True):
    if which == "all":
        raise ValueError("Must specify if star is a single or binary.")

    te = []
    l = []

    # assume the star exists at all times
    for time in times:
        bdata, sdata = init_data(path, time, which, clean)

        if which == "singles":
            names = list(sdata["name"])
            i = names.index(name)
            te.append(sdata["te"][i])
            l.append(sdata["zl"][i])
        else:
            names = list(bdata["name1"])
            try:
                i = names.index(name)
                te.append(bdata["te1"][i])
                l.append(bdata["zl1"][i])
            except ValueError:
                # not found, try second member
                names = list(bdata["name2"])
                i = names.index(name)
                te.append(bdata["te2"][i])
                l.append(bdata["zl2"][i])

    plt.scatter(te, l, marker=".", s=10, c=1e-3 * t_star * times)
    plt.xlabel(r"$\log_{10} T_{eff}$")
    plt.ylabel(r"$\log_{10} L / L_{\odot}$")

    plt.xlim(reversed(plt.xlim()))

    plt.title(f"Path of Star {name} at different times")

    plt.colorbar(label="t / Gyr")

    if save:
        save_plot(clean, which, None, "path")
        np.save("te.npy", te)
        np.save("l.npy", l)



def radial_test(path, time, clean=True, save=True):
    bdata, sdata = init_data(path, time, which="all", clean=clean)

    print(bdata["name1"].shape, bdata["ri"].shape)

    bri = np.array(bdata["ri"]) / 100
    # bri = np.array(bdata["ri"])

    x = np.linspace(0, len(sdata["time"]), len(sdata["time"]))

    plt.scatter(x, sdata["ri"], color="C0", marker=".", s=0.5)

    x = np.linspace(0, len(sdata["time"]), len(bdata["name1"]))

    plt.scatter(x, bri, color="C1", marker=".", s=0.5)
    plt.scatter(x, bri, color="C1", marker=".", s=0.5)

    plt.yscale("log")


    if save:
        save_plot(clean, "all", time, "rtest")



def main(argv: list) -> int:

    path = Path("../reconstruction/output")

    # use a bigger font size
    matplotlib.rcParams.update({"font.size": 20})

    # use large figures
    matplotlib.rcParams.update({"figure.figsize": (10, 9)})


    # radial_test(path, 0)
    # plt.close()

    # times = [10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 10000, 20000, 30000, 40000, 53330]

    # for time in times:
    #     mass_seg(path, time)
    #     plt.close()

    # HRD
    # times = [0, 53330]

    # for time in times:
    #     hrd(path, time, which="singles", clean=False)
    #     plt.close()
    #     # stellar_types(path, time)
    #     # plt.close()
    #     radial_hist(path, time)
    #     plt.close()
    #     mass_dist(path, time)
    #     plt.close()
    #     mass_seg(path, time)
    #     plt.close()


    # which = ["all", "singles", "binaries"]
    # clean = [True, False]
    # for time in times:
    #     for w in which:
    #         for c in clean:
    #             hrd(path, time, which=w, clean=c, save=True)
    #             plt.close()

    # binary fraction over time

    # only every 10th file was reconstructed
    # times = 10 * np.linspace(0, 5333, 400, dtype=int)

    # fig = binary_fraction(path, times)
    # plt.close(fig)
    
    # times = 10 * np.linspace(0, 5333, 10, dtype=int)
    # for time in times:
    #     radial_cfractions(path, time)
    #     plt.close()
    #     radial_remnants(path, time)
    #     plt.close()

    # times = 10 * np.linspace(0, 5333, 200, dtype=int)
    # lagrange_radii(path, times, fraction=0.01)



    # find eligible stars to follow

    # _, sdata = init_data(path, 0)

    # kw0 = np.array(sdata["kw"])
    # name0 = np.array(sdata["name"])
    # teff0 = np.array(sdata["te"])

    # crit = teff0 < 3.78

    # kw0, name0 = kw0[crit], name0[crit]

    # sort = np.argsort(name0)
    # kw0, name0 = kw0[sort], name0[sort]

    # crit = kw0 == 1

    # kw0, name0 = kw0[crit], name0[crit]

    # _, sdata = init_data(path, 53330)

    # kw1 = np.array(sdata["kw"])
    # name1 = np.array(sdata["name"])
    # teff1 = np.array(sdata["te"])

    # crit = teff1 < 3.78
    # kw1, name1 = kw1[crit], name1[crit]

    # sort = np.argsort(name1)
    # kw1, name1 = kw1[sort], name1[sort]

    # crit = kw1 == 11

    # kw1, name1 = kw1[crit], name1[crit]


    # survived = np.array([i for i in name0 if i in name1])

    # for name in survived:
    #     i0 = np.argmax(name0 == name)
    #     i1 = np.argmax(name1 == name)

    #     print(kw0[i0], name0[i0], kw1[i1], name1[i1])


    # manually select a star and follow it

    name = 30500
    times = 10 * np.linspace(0, 5333, 5000, dtype=int)

    # for time in times:
    #     p = path / Path("sev.83_" + str(time))
    #     if not p.is_file():
    #         fetch_single(p)

    follow_star_hrd(path, times, name, clean=False)

    return 0


if __name__ == "__main__":
    import sys
    main(sys.argv)
