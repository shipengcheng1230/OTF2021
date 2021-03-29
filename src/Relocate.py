import copy
import itertools
import json
import multiprocessing as mp
import operator
import os
import shutil
from collections import deque, namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from copy import copy
from datetime import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import networkx as nx
import nlopt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from obspy import read
from obspy.imaging.beachball import beach
from obspy.io.sac.util import SacIOError
from obspy.signal.cross_correlation import correlate, xcorr_max
from pyproj import Geod
from scipy.optimize import curve_fit, least_squares, minimize

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

    def bootstrap_uncertainty(x, y, p0, numiters=400, stderr=3.75, nsigma=1.0):
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

    fm1, fm2 = fa.fm[str(id1)]["np1"], fa.fm[str(id2)]["np1"]
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

        if st1[0].stats["sampling_rate"] != st2[0].stats["sampling_rate"]:
            maxSamplingRate = max(st1[0].stats["sampling_rate"], st2[0].stats["sampling_rate"])
            for ss in [st1, st2]:
                if maxSamplingRate != ss[0].stats["sampling_rate"]:
                    ss.resample(maxSamplingRate, no_filter=True)

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
        json.dump(summary, fp, indent=4)

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
                                  f"{outlierCC:.2f})", f"Observation (cc $<$ {outlierCC:.2f})"], loc="lower center", bbox_to_anchor=(0.5, -0.3))

        l1, = ax2.plot([], [], color="tab:pink", linestyle="-")
        l2, = ax2.plot([], [], color="gray", linestyle="--")
        l3 = ax2.scatter([], [], s=20, marker="*", c="orange")
        ax2.legend([l1, l2, l3], ["$\mathrm{cc} \geq \;$"+f"{outlierCC:.2f}", "$\mathrm{cc} <\;$" +
                                  f"{outlierCC:.2f}", fa.name], loc="lower left", bbox_to_anchor=(-0.60, 0.0))
        fig.savefig(nameBase + ".pdf")
        print(f"Saved {os.path.basename(nameBase)}.")
        plt.close(fig)

    plotCosineFit()


def crossCorrelate(fa):
    dfPairss = pd.read_csv(os.path.join(fa.dir, "catalog-pair.csv"))
    futures = []

    ccFiles = [os.path.join(fa.ccDir, x) for x in os.listdir(fa.ccDir)]
    for f in ccFiles:
        os.remove(f)
    # for r in dfPairss.itertuples(index=True):
    #     print(f"{r.Index + 1}/{dfPairss.shape[0]} ...")
    #     cc(fa, r.id1, r.t1, r.lat1, r.lon1, r.mag1, r.id2, r.t2, r.lat2, r.lon2, r.mag2)
    with ProcessPoolExecutor(max_workers=20) as executor:
        for r in dfPairss.itertuples(index=True):
            futures.append(executor.submit(cc, fa, r.id1, r.t1, r.lat1,
                                           r.lon1, r.mag1, r.id2, r.t2, r.lat2, r.lon2, r.mag2))
        [future.result() for future in as_completed(futures)]


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


def optimize(fa):

    def formEdge(x: dict, linkNum: int = 8):
        return \
            (not np.isnan(x['B'])) and \
            (not np.isinf(x['dB'])) and \
            x['numHighCC'] >= linkNum

    def buildPairGraph():
        dfPairs = pd.read_csv(os.path.join(fa.dir, "catalog-pair.csv"))
        files = (x for x in os.listdir(fa.ccDir) if x.endswith(".json"))
        pairkey2row = {(r.id1, r.id2): r for r in dfPairs.itertuples()}

        G = nx.Graph()
        for file in files:
            with open(os.path.join(fa.ccDir, file), "r") as fp:
                data = json.load(fp)
                if formEdge(data):
                    info = os.path.splitext(file)[0].split('-')
                    id1, id2 = map(str, info)
                    r = pairkey2row.get((id1, id2), None)
                    if not r:
                        continue
                    if id1 != r.id1 or id2 != r.id2:
                        raise ValueError("pair id mismatch.")

                    # duplicate hashing is ignored
                    G.add_node(id1,
                               lat=r.lat1, lon=r.lon1, mag=r.mag1,
                               t=r.t1, group=-1)
                    G.add_node(id2,
                               lat=r.lat2, lon=r.lon2, mag=r.mag2,
                               t=r.t2, group=-1)
                    G.add_edge(id1, id2,
                               B=data['B'], C=data['C'],
                               dB=data['dB'], dC=data['dC'])
        return G

    def traverseGraph(G, relocateOneWay=True):
        # BFS, relocate one group of events by single reference abiding shortest path
        # This, however, does not account for discrepancy with direct observation and
        # new locations. Using optimization is perferred, see below.

        # If not relocate, it aims to idenfity groups and label accordingly.
        def traverse(n):
            queueSet.add(n)
            for k in G.adj[n].keys():
                if (not visited[k]) and (k not in queueSet):
                    if relocateOneWay:
                        B, C = G.adj[n][k]['B'], G.adj[n][k]['C']
                        # all pairs id1 is more recent than id2
                        C = C if IDTimePair[n] > IDTimePair[k] else C + 180.0  # only change `B` or `C`, not both
                        lat1, lon1 = G.nodes.data(
                            'lat')[n], G.nodes.data('lon')[n]
                        lon2, lat2, _ = geod.fwd(lon1, lat1, C, B*1e3)
                        G.nodes[k].update(lat=lat2, lon=lon2)
                    visited[k] = True
                    IDTimePairDynamic.pop(k, None)
                    G.nodes[k].update(group=groupid)
                    Q.append(k)
                    queueSet.add(k)

        geod = Geod(ellps="WGS84")
        Q = deque()
        queueSet = set()
        visited = dict(G.nodes.data('visited'))
        IDTimePair = dict(G.nodes.data("t"))
        IDTimePairDynamic = dict(G.nodes.data("t"))
        groupid = 1

        while count := sum([1 for (k, v) in visited.items() if not v]) > 0:
            # use newest event as reference
            startID = max(IDTimePairDynamic, key=IDTimePairDynamic.get)
            Q.append(startID)
            visited[startID] = True
            IDTimePairDynamic.pop(startID)
            G.nodes[startID].update(group=groupid)
            while Q:
                traverse(Q.popleft())
            groupid += 1
        return G

    def traverseOptimizeGraph(G):
        Q = deque()
        geod = Geod(ellps="WGS84")
        optimized = dict(G.nodes.data('visited'))
        queueSet = set()
        IDTimePair = dict(G.nodes.data("t"))
        IDTimePair2 = dict(G.nodes.data("t"))

        def optimizeNodeLocation(node):

            def objective(x, lons, lats, B, C, dB, dC):
                res = 0.0
                for i in range(len(lons)):
                    fz, _, dist = geod.inv(x[0], x[1], lons[i], lats[i])
                    fz, dist = _normalize_fz_dist(fz, dist)
                    res += np.abs(angularDiff(fz, C[i]) / dC[i]) + \
                        np.abs((dist/1e3 - B[i]) / dB[i])
                return res

            ks = G.adj[node].keys()
            queueSet.add(node)
            IDTimePair.pop(node, None)
            for k in ks:
                if (not optimized[k]) and (k not in queueSet):
                    Q.append(k)
                    queueSet.add(k)
                    IDTimePair.pop(k, None)

            lons = np.array([G.nodes.data('lon')[k] for k in ks])
            lats = np.array([G.nodes.data('lat')[k] for k in ks])
            x0 = np.array([angularMean(lons), angularMean(lats)])
            Bs = np.array([G.adj[node][k]['B'] for k in ks])
            Cs = np.array([G.adj[node][k]['C'] for k in ks])
            dBs = np.array([G.adj[node][k]['dB'] for k in ks])
            dCs = np.array([G.adj[node][k]['dC'] for k in ks])
            for i, k in enumerate(ks):
                # all pairs id1 is more recent than id2
                if IDTimePair2[node] < IDTimePair2[k]:  # correction for direction
                    Cs[i] += 180.0
                if Bs[i] < 0:  # correction for negative distance
                    Bs[i] *= -1
                    Cs[i] += 180.0
                Cs[i] %= 360  # correction for azimuth angle between [0, 360)
            res = minimize(objective, x0, args=(
                lons, lats, Bs, Cs, dBs, dCs), method="Nelder-Mead")
            G.nodes[node].update(lat=res.x[1], lon=res.x[0])
            optimized[node] = True

        while count := sum([1 for (k, v) in optimized.items() if not v]) > 0:
            # startID = min([k for (k, v) in optimized.items() if not v])
            startID = max(IDTimePair, key=IDTimePair.get)
            Q.append(startID)
            IDTimePair.pop(startID)
            while Q:
                optimizeNodeLocation(Q.popleft())
        return G

    def optimizeLocation(G, relocateTwoWay=True, relocateTwoWayIter=5, relocateGlobal=True):
        if relocateTwoWay:
            for _ in range(relocateTwoWayIter):
                traverseOptimizeGraph(G)

        data = {k: G.nodes[k] for k in G.nodes}
        uniqueGroupID = set([data[k]['group'] for k in data.keys()])
        linkContent = {x: [] for x in [
            "id1", "t1", "lat1", "lon1", "mag1",
            "id2", "t2", "lat2", "lon2", "mag2",
            "dist", "B", "dB",
            "azi", "C", "dC",
            "group", "weight",
        ]}
        geod = Geod(ellps="WGS84")
        IDTimePair = dict(G.nodes.data("t"))

        for gid in uniqueGroupID:
            es = [x for x in G.edges if data[x[0]]['group'] == gid]
            es.sort(key=lambda x: (IDTimePair[x[0]], IDTimePair[x[1]]), reverse=True)
            pks = list(set(itertools.chain(*es)))
            pks.sort(key=lambda x: IDTimePair[x], reverse=True)
            id2t = {k: IDTimePair[k] for k in pks}
            masterId = max(id2t, key=id2t.get)
            masterLat, masterLon = data[masterId]['lat'], data[masterId]['lon']
            key2id = {pks[i]: i for i in range(len(pks))}
            Bs = [G.edges[x]['B'] for x in es]
            Cs = [G.edges[x]['C'] for x in es]
            dBs = [G.edges[x]['dB'] for x in es]
            dCs = [G.edges[x]['dC'] for x in es]
            lat0 = np.array([data[x]['lat'] for x in pks])
            lon0 = np.array([data[x]['lon'] for x in pks])
            u0 = [[lat0[i], lon0[i]] for i in range(len(pks))]
            u0 = list(itertools.chain(*u0))
            u0.append(1)  # initial weight

            if not relocateGlobal:
                x = u0
            else:
                avglat0, avglon0 = angularMean(lat0), angularMean(lon0)
                searchRange = 10.0  # within 10 degree, should be sufficient
                # not rigorously right,
                # but we don't have events at polar region or date changing line
                lblat = np.clip(lat0 - searchRange, -90, 90)
                ublat = np.clip(lat0 + searchRange, -90, 90)
                lblon = np.clip(lon0 - searchRange, -180, 180)
                ublon = np.clip(lon0 + searchRange, -180, 180)

                lb = [[x, y] for x, y in zip(lblat, lblon)]
                lb = list(itertools.chain(*lb))
                lb.append(0.2)  # lower bound for relative weight
                ub = [[x, y] for x, y in zip(ublat, ublon)]
                ub = list(itertools.chain(*ub))
                ub.append(2)  # upper bound for relative weight

                objective, masterLatitudeConstraint, masterLongitudeConstraint = \
                    objectiveFuncFactory(
                        masterId, masterLat, masterLon, Bs, Cs, dBs, dCs, key2id, es)

                opt = nlopt.opt(nlopt.LN_COBYLA, 2*len(pks)+1)
                opt.set_min_objective(objective)
                opt.add_equality_constraint(masterLatitudeConstraint, 1e-6)
                opt.add_equality_constraint(masterLongitudeConstraint, 1e-6)
                opt.set_lower_bounds(lb)
                opt.set_upper_bounds(ub)
                opt.set_xtol_rel(1e-6)
                x = opt.optimize(u0)

            for e in es:
                _add_link_content(linkContent, G, e, x, key2id, geod, IDTimePair)

        pd.DataFrame(linkContent).to_csv(os.path.join(
            fa.dir, "catalog-link.csv"), index=False)
        content = {'id': [], 'time': [], 'lat': [],
                   'lon': [], 'mag': [], 'group': []}
        content['id'].extend(linkContent['id1'])
        content['id'].extend(linkContent['id2'])
        content['time'].extend(linkContent['t1'])
        content['time'].extend(linkContent['t2'])
        content['lat'].extend(linkContent['lat1'])
        content['lat'].extend(linkContent['lat2'])
        content['lon'].extend(linkContent['lon1'])
        content['lon'].extend(linkContent['lon2'])
        content['mag'].extend(linkContent['mag1'])
        content['mag'].extend(linkContent['mag2'])
        content['group'].extend(linkContent['group'])
        content['group'].extend(linkContent['group'])
        pd.DataFrame(content).drop_duplicates(subset=['id'], keep='last').sort_values(
            by="time").to_csv(os.path.join(fa.dir, "catalog-relocated.csv"), index=False)

    def _add_link_content(linkContent, G, e, sol, key2id, geod, IDTimePair):
        # repeated code
        # all pairs id1 is more recent than id2
        if IDTimePair[e[0]] < IDTimePair[e[1]]:
            e = (e[1], e[0])
        id1, id2 = key2id[e[0]], key2id[e[1]]

        lat1, lon1 = sol[2*id1], sol[2*id1+1]
        lat2, lon2 = sol[2*id2], sol[2*id2+1]

        linkContent['id1'].append(e[0])
        linkContent['t1'].append(G.nodes[e[0]]['t'])
        linkContent['lat1'].append(lat1)
        linkContent['lon1'].append(lon1)
        linkContent['mag1'].append(G.nodes[e[0]]['mag'])
        linkContent['id2'].append(e[1])
        linkContent['t2'].append(G.nodes[e[1]]['t'])
        linkContent['lat2'].append(lat2)
        linkContent['lon2'].append(lon2)
        linkContent['mag2'].append(G.nodes[e[1]]['mag'])
        fz, dist = _objection_val(sol, e, key2id, geod)
        linkContent['dist'].append(dist/1e3)
        linkContent['azi'].append(fz)
        bb, cc = G.edges[e]['B'], G.edges[e]['C'] % 360
        cc, bb = _normalize_fz_dist(cc, bb)
        linkContent['B'].append(bb)
        linkContent['C'].append(cc)
        linkContent['dB'].append(G.edges[e]['dB'])
        linkContent['dC'].append(G.edges[e]['dC'] % 360)
        linkContent['group'].append(G.nodes[e[0]]['group'])
        linkContent['weight'].append(sol[-1])

    def mergeCatalogue():
        content_strs = ['id', 'time', 'lat', 'lon', 'mag', 'group']
        content = {x: [] for x in content_strs}
        dforg = pd.read_csv(os.path.join(fa.dir, "catalog.csv"))
        dfcc = pd.read_csv(os.path.join(fa.dir, "catalog-relocated.csv"))
        relocatedID = set(dfcc["id"].to_list())
        for r in dfcc.itertuples():
            content['id'].append(r.id)
            content['time'].append(r.time)
            content['lat'].append(r.lat)
            content['lon'].append(r.lon)
            content['mag'].append(r.mag)
            content['group'].append(r.group)
        for r in dforg.itertuples():
            if r.id not in relocatedID:
                mw = mag2mw(r.mag, r.magType)
                content['id'].append(r.id)
                content['time'].append(r.time)
                content['lat'].append(r.lat)
                content['lon'].append(r.lon)
                content['mag'].append(mw)
                content['group'].append(-1)
        pd.DataFrame(content).sort_values(by=["time"]).to_csv(
            os.path.join(fa.dir, "catalog-merged.csv"), index=False)

    def _objection_val(sol, e: list, key2id: dict, geod):
        id1, id2 = key2id[e[0]], key2id[e[1]]
        lat1, lon1 = sol[2*id1], sol[2*id1+1]
        lat2, lon2 = sol[2*id2], sol[2*id2+1]
        fz, _, dist = geod.inv(lon1, lat1, lon2, lat2)
        fz, dist = _normalize_fz_dist(fz, dist)
        return fz, dist

    def _normalize_fz_dist(fz: float, dist: float):
        if dist < 0:
            dist *= -1
            fz += 180.0
        fz %= 360
        return fz, dist

    def objectiveFuncFactory(masterID, masterLat, masterLon, Bs, Cs, dBs, dCs, key2id, es, weight=1.0):
        # weight denotes relative contribution between `dist` and `azi`
        geod = Geod(ellps="WGS84")
        _B = copy.copy(Bs)
        _C = list(map(lambda x: x % 360, Cs))
        _dB = copy.copy(dBs)
        _dC = list(map(lambda x: x % 360, dCs))
        for i in range(len(_B)):
            _C[i], _B[i] = _normalize_fz_dist(_C[i], _B[i])
            _dC[i], _dB[i] = _normalize_fz_dist(_dC[i], _dB[i])
            if _dC[i] > 180:
                _dC[i] = 360.0 - _dC[i]

        _dB2 = [x**2 for x in _dB]
        _dC2 = [x**2 for x in _dC]

        def objective(x, grad):
            # x = [lat_1, lon_1, lat_2, lon_2, ..., lat_n, lon_n, weight]
            if grad.size > 0:
                raise RuntimeError("Use gradient free method!")

            y = 0.0
            for i, e in enumerate(es):
                fz, dist = _objection_val(x, e, key2id, geod)
                y += np.abs((dist/1e3 - _B[i]) / _dB[i]) + \
                    x[-1] * np.abs(angularDiff(fz, _C[i]) / _dC[i])
            return np.sqrt(y)

        def masterLatitudeConstraint(x, grad):
            if grad.size > 0:
                raise RuntimeError("Use gradient free method!")
            return x[2 * key2id[masterID]] - masterLat

        def masterLongitudeConstraint(x, grad):
            if grad.size > 0:
                raise RuntimeError("User gradient free method!")
            return x[2 * key2id[masterID] + 1] - masterLon

        return objective, masterLatitudeConstraint, masterLongitudeConstraint

    G = buildPairGraph()
    G = traverseGraph(G, True)
    optimizeLocation(G, relocateTwoWay=True, relocateGlobal=False)
    mergeCatalogue()

if __name__ == "__main__":

    names = dfFaults["Name"].to_list()
    for name in names:
        print(name)
        f = RelocationProcedure(name.strip())
        crossCorrelate(f)
        optimize(f)
