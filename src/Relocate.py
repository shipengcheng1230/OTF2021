import json
import multiprocessing as mp
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, wait, as_completed
from copy import copy
from collections import namedtuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from obspy import read
from obspy.signal.cross_correlation import correlate, xcorr_max
from pyproj import Geod
from scipy.optimize import curve_fit, least_squares
from obspy.imaging.beachball import beach
from obspy.io.sac.util import SacIOError

from GetData import AbstractFaultProcess
from utils import *


def cc(fa, id1, t1, lat1, lon1, mag1, id2, t2, lat2, lon2, mag2):
    # cannot serialize pandas frame, which sliently quits multiprocessor

    outlierCC = fa.relocationConfig.get("outlierCC", 0.70)
    outlierDt = fa.relocationConfig.get("outlierDt", 35)
    numCCToFit = fa.relocationConfig.get("numCCToFit", 8)
    vgroup = fa.relocationConfig.get("vgroup", 3.75)
    outlierDx = vgroup * outlierDt
    geod = Geod(ellps="WGS84")

    def badCoverage(azi, maxgap=120.0):  # exclude those with azimuth gap larger than `maxgap`
        if len(azi) == 0:
            return False
        _azi = sorted([x % 360 for x in azi])
        diff = [_azi[i+1] - _azi[i] for i in range(len(_azi) - 1)]
        diff.append(360 - _azi[-1] + _azi[0])
        return np.any(np.array(diff) > maxgap)

    def residual_cosine(x, azi, dt):
        return x[0] + x[1] * np.cos(np.deg2rad(azi - x[2])) - dt

    def fit_cosine_ls_norm(azi, dx, cc, u0):
        mask = np.logical_and(cc >= outlierCC, np.abs(dx) <= outlierDx)
        _azi = azi[mask]
        _dx = dx[mask]
        if len(_azi) < numCCToFit or badCoverage(_azi):
            return np.array([np.nan, np.nan, np.nan]), np.array([np.inf, np.inf, np.inf])

        # initial solution affect the local minimum obtained
        res = least_squares(residual_cosine, [0., u0, 0.], args=(
            _azi, _dx), loss='soft_l1', jac='3-point')

        # On way of uncertainty, but inappropriate since we use other forms of norm
        # https://stackoverflow.com/a/21844726/8697614
        # J = res.jac
        # cov = np.linalg.inv(J.T.dot(J))
        # pcov = cov * (res.fun ** 2).sum() / (len(_azi) - len(res.x)) # reduced chi-square distribution
        # err = np.sqrt(np.diagonal(pcov))

        # Another way of uncertainty, still assume your function value is chi-square
        # https://stackoverflow.com/a/53489234/8697614, notice `dx[2] can exceed 360`
        # J = res.jac
        # cov = np.linalg.inv(J.T.dot(J))
        # err2 = np.sqrt(np.diagonal(cov) * np.abs(res.cost))
        # return res.x, err

        # The third way: bootstrap
        # This is what McGuire did, assuming 1s error in dt
        # err = bootstrap_uncertainty(_azi, _dx, [0., u0, 0.], stderr=1.0 * vgroup)
        # we instead use fitting residual as the error for bootstrap
        err = bootstrap_uncertainty(_azi, _dx, [0., u0, 0.], stderr=np.std(
            residual_cosine(res.x, _azi, _dx)))
        return res.x, err

    def bootstrap_uncertainty(x, y, p0, numiters=200, stderr=3.75, nsigma=1.0):
        x_cmt, x_dist, x_azi = [], [], []
        n = len(y)
        for i in range(numiters):
            yy = y + np.random.default_rng().normal(0.0, stderr, n)
            res = least_squares(residual_cosine, p0, args=(
                x, yy), loss='soft_l1', jac='3-point')
            x_cmt.append(res.x[0])
            x_dist.append(res.x[1])
            x_azi.append(res.x[2])
        return nsigma*np.std(x_cmt), nsigma*np.std(x_dist), nsigma*np.std(x_azi)

    fm1, fm2 = fa.fm[id1]["np1"], fa.fm[id2]["np1"]
    wv1, wv2 = [], []
    content = {x: [] for x in ["azi", "dt",
                               "cc", "net", "sta", "staLat", "staLon"]}
    nameBase = os.path.join(
        fa.ccDir, '-'.join(map(str, [id1, id2])))
    dfStations = pd.read_csv(os.path.join(fa.dir, "stations.csv"))
    for rs in dfStations.itertuples():
        waveName1 = str(id1) + '-' + rs.net + '-' + rs.sta + '.sac'
        waveName2 = str(id2) + '-' + rs.net + '-' + rs.sta + '.sac'
        wavePath1 = os.path.join(fa.waveDir, waveName1)
        wavePath2 = os.path.join(fa.waveDir, waveName2)

        if (
            np.isnan(rs.dist) or
            not os.path.isfile(wavePath1) or
            not os.path.isfile(wavePath2) or
            os.path.getsize(wavePath1) == 0 or
            os.path.getsize(wavePath2) == 0
        ):
            continue

        midLon, midLat = angularMean(
            [lon1, lon2]), angularMean([lat1, lat2])
        fz, _, _ = geod.inv(midLon, midLat, rs.lon,
                            rs.lat)  # fz: (-180, 180)
        try:
            st1 = read(wavePath1, format="SAC")
        except SacIOError:
            os.remove(wavePath1)
            continue
        try:
            st2 = read(wavePath2, format="SAC")
        except SacIOError:
            os.remove(wavePath2)
            continue

        cc = correlate(
            st1[0], st2[0], demean=True, normalize='naive',
            shift=min(st1[0].stats.npts, st2[0].stats.npts, int(
                np.round(outlierDt / st1[0].stats.delta * 2.0)))
        )
        # consider only positive cc
        shift, value = xcorr_max(cc, abs_max=False)
        dt = st1[0].stats.delta * shift
        content['azi'].append(fz)
        content['dt'].append(dt)
        content['cc'].append(value)
        content['net'].append(rs.net)
        content['sta'].append(rs.sta)
        content['staLat'].append(rs.lat)
        content['staLon'].append(rs.lon)
        wv1.append(st1[0].data)
        wv2.append(st2[0].data)

    pd.DataFrame(content).to_csv(nameBase + ".csv", index=False)
    if len(content["dt"]) < numCCToFit:
        return

    yy = np.array(content['dt']) * vgroup
    xx = np.array(content['azi'])
    # initial guess of distance, important for least square fit
    _, _, u0 = geod.inv(lon1, lat1, lon2, lat2)
    popt, pcov = fit_cosine_ls_norm(
        xx, yy, np.array(content['cc']), np.abs(u0 / 1e3))
    A, B, C = popt
    dA, dB, dC = pcov
    highcc = list(filter(lambda x: np.abs(x) >= outlierCC, content['cc']))
    summary = {
        'A': popt[0], 'B': popt[1], 'C': popt[2],
        'dA': pcov[0], 'dB': pcov[1], 'dC': pcov[2],
        'numHighCC': len(highcc),
    }
    with open(nameBase + ".json", "w") as fp:
        json.dump(summary, fp)

    def plotCosineFit():
        fig = plt.figure(figsize=(8, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[2, 1])
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(
            gs[1, 0], projection=ccrs.AzimuthalEquidistant(lon1, lat1))
        ax2.stock_img()
        ax2.scatter(lon1, lat1, s=20, marker="*", c="orange",
                    transform=ccrs.PlateCarree(), zorder=2)

        ax3 = fig.add_axes([0.55, 0.89, 0.1, 0.1])
        ax3.set_aspect("equal")
        if np.isnan(fm1[0]):
            bc1 = beach([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], facecolor="white",
                        edgecolor="lightgray", xy=(0.5, 0.75), width=0.4)
        else:
            bc1 = beach(fm1, facecolor="tab:blue", xy=(0.5, 0.75), width=0.4)
        if np.isnan(fm2[0]):
            bc2 = beach([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], facecolor="white",
                        edgecolor="lightgray", xy=(0.5, 0.25), width=0.4)
        else:
            bc2 = beach(fm2, facecolor="tab:red", xy=(0.5, 0.25), width=0.4)
        ax3.add_collection(bc1)
        ax3.add_collection(bc2)
        ax3.axis('off')

        for i in range(len(wv1)):
            # do not set `step=delta` which may cause length mismatch
            y1 = np.arange(0.0, len(wv1[i])) * st1[0].stats.delta
            y2 = np.arange(0.0, len(wv2[i])) * st2[0].stats.delta
            w1 = wv1[i] - np.nanmean(wv1[i])
            w2 = wv2[i] - np.nanmean(wv2[i])
            w1 = w1 / np.nanmax(np.abs(w1)) * 8
            w2 = w2 / np.nanmax(np.abs(w2)) * 8

            isoutlier = not (np.abs(content["dt"][i]) <= outlierDt and np.abs(
                content["cc"][i]) >= outlierCC)
            linestyle = '--' if isoutlier else '-'
            alpha = 0.2 if isoutlier else 1.0
            fc = 'white' if isoutlier else 'tab:green'
            ax0.plot(y1, w1 + content["azi"][i], color="tab:blue",
                     linestyle=linestyle, alpha=alpha, clip_on=False)
            ax0.plot(y2, w2 + content["azi"][i], color="tab:red",
                     linestyle=linestyle, alpha=alpha, clip_on=False)
            ax1.scatter(content["dt"][i], content["azi"][i],
                        fc=fc, s=15, marker="o", ec="black", zorder=2)

            trajectorColor = "gray" if isoutlier else "tab:pink"
            trajectorStyle = "--" if isoutlier else "-"
            ax2.plot([lon1, content["staLon"][i]], [lat1, content["staLat"][i]],
                     color=trajectorColor, transform=ccrs.Geodetic(), zorder=1, linestyle=trajectorStyle)

        xx = np.arange(-180, 180, 1.0)
        if np.isnan(A):
            yy = np.array([np.nan for _ in xx])
            fittingColor = "gray"
        else:
            yy = A + B * np.cos(np.deg2rad(xx - C))
            yy /= vgroup
            fittingColor = "tab:orange"

        ax1.plot(yy, xx, c=fittingColor, zorder=1)
        _b, _c = B, C
        if _b < 0:
            _b *= -1
            _c += 180
        _c %= 360
        ax1.text(0.5, 1.05, "$\Delta x = \;$" + f"{_b:.0f}" + "$\;\pm \;$" + f"{dB:.0f}" + "$\;\mathrm{km}$",
                 ha='center', va='center', transform=ax1.transAxes, backgroundcolor="steelblue", color="white")
        ax1.text(0.5, 1.12, r"$\theta = \;$" + f"{_c:.1f}" + "$\;\pm \;$" + f"{dC:.1f}" + "$^{\circ}$",
                 ha='center', va='center', transform=ax1.transAxes, backgroundcolor="steelblue", color="white")
        ax0.set_ylim([-180, 180])
        ax1.set_ylim([-180, 180])
        ax1.set_xlim([-outlierDt, outlierDt])
        ax1.set_yticklabels([])
        ax1.grid(True, which="major", color='gray',
                 linestyle='--', linewidth=1)
        ax1.set_xlabel("$\Delta T$ (second)")
        ax0.set_xlabel("Time (second)")
        ax0.set_ylabel("Azimuth (deg)")
        ax0.autoscale(enable=True, axis='x', tight=True)

        l1, = ax0.plot([], [], color="tab:blue")
        l2, = ax0.plot([], [], color="tab:red")
        _tstr1 = datetime.fromisoformat(t1)
        _tstr1 = _tstr1.strftime("%Y-%m-%d %H:%M:%S"+f'  M{mag1:.1f}')
        _tstr2 = datetime.fromisoformat(t2)
        _tstr2 = _tstr2.strftime("%Y-%m-%d %H:%M:%S"+f'  M{mag2:.1f}')
        ax0.legend([l1, l2], [_tstr1, _tstr2],
                   loc="upper center", bbox_to_anchor=(0.5, 1.18))

        l1, = ax1.plot([], [], color="tab:orange")
        l2 = ax0.scatter([], [], fc='tab:green', s=15, marker="o", ec="black")
        l3 = ax0.scatter([], [], fc='white', s=15, marker="o", ec="black")
        ax1.legend([l1, l2, l3], ["Cosine Fitting", "Observation (cc $\geq \;$" +
                                  f"{outlierCC:.2f})", f"Observation (cc $>$ {outlierCC:.2f})"], loc="lower center", bbox_to_anchor=(0.5, -0.3))

        l1, = ax2.plot([], [], color="tab:pink", linestyle="-")
        l2, = ax2.plot([], [], color="gray", linestyle="--")
        l3 = ax2.scatter([], [], s=20, marker="*", c="orange")
        ax2.legend([l1, l2, l3], ["$\mathrm{cc} \geq \;$"+f"{outlierCC:.2f}", "$\mathrm{cc} <\;$" +
                                  f"{outlierCC:.2f}", fa.name], loc="lower left", bbox_to_anchor=(-0.60, 0.0))
        fig.savefig(nameBase + ".pdf")
        print(f"Saved {nameBase}.")
        plt.close(fig)

    plotCosineFit()


class RelocationProcedure(AbstractFaultProcess):

    def __init__(self, name, *args, **kwargs):
        super().__init__(name)
        self.waveDir = os.path.join(self.dir, "waves")
        if not os.path.isdir(self.waveDir):
            os.mkdir(self.waveDir)
        self.ccDir = os.path.join(self.dir, "cc")
        if not os.path.isdir(self.ccDir):
            os.mkdir(self.ccDir)
        with open(os.path.join(self.dir, "mt.json")) as fp:
            self.fm = json.load(fp)
        self.relocationConfig = kwargs


def crossCorrelate(fa):
    dfPairs = pd.read_csv(os.path.join(fa.dir, "catalog-pair.csv"))
    futures = []
    # for r in dfPairs.itertuples(index=True):
    #     print(f"{r.Index + 1}/{dfPairs.shape[0]} ...")
    #     cc(fa, r.id1, r.t1, r.lat1, r.lon1, r.mag1, r.id2, r.t2, r.lat2, r.lon2, r.mag2)
    with ProcessPoolExecutor(max_workers=20) as executor:
        for r in dfPairs.itertuples(index=True):
            futures.append(executor.submit(cc, fa, r.id1, r.t1, r.lat1,
                                           r.lon1, r.mag1, r.id2, r.t2, r.lat2, r.lon2, r.mag2))
        [future.result() for future in as_completed(futures)]


def optimize(self):
    pass


f = RelocationProcedure("Discovery")
crossCorrelate(f)
