import json
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from datetime import datetime

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import netCDF4
import numpy as np
import numpy.ma as ma
import pandas as pd
from cartopy.io import LocatedImage, PostprocessedRasterSource
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib import gridspec
from matplotlib.colors import LightSource, rgb2hex
from matplotlib.legend_handler import HandlerPatch
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.ticker import NullFormatter
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from obspy.imaging.beachball import beach
from pyproj import Geod
from scipy import interpolate, stats

from GetData import AbstractFaultProcess
from scalebar import dist_func_complete, scale_bar
from utils import *

warnings.filterwarnings("ignore")

GMRT_BATHY_GRID = "/mnt/disk6/pshi/otfloc2/bathy"


class PlotProcedure(AbstractFaultProcess):

    def __init__(self, name):
        super().__init__(name)
        with open(os.path.join(self.dir, "mt.json")) as fp:
            self.fm = json.load(fp)

        for i in range(2, 5):
            self.config["adjust"].setdefault(str(i), {"lat": 0.0, "lon": 0.0})


    def plotTimeSpaceLayout(self, overWrite=True):
        output = os.path.join(self.dir, "layout.pdf")

        if not overWrite and os.path.isfile(output):
            return

        clipped_date_post1950 = datetime(1949, 6, 1)
        clipped_date_post1995 = datetime(1994, 12, 31)
        clipped_date_post1990 = datetime(1989, 12, 31)

        rotatediff = 90 - self.df["strike"].iloc[0] % 180
        if rotatediff > 0:
            rotlon0 = self.df["Longitude"].iloc[0] - 90
            rotlat0 = 90 - rotatediff
        else:
            rotlon0 = self.df["Longitude"].iloc[0] + 90
            rotlat0 = 90 + rotatediff
        rotatedpole = ccrs.RotatedPole(rotlon0, rotlat0, 0.0)

        dfMerged = pd.read_csv(os.path.join(self.dir, "catalog-merged.csv"))

        figsize = (10, 8)
        fig = plt.figure(figsize=figsize)
        grid = gridspec.GridSpec(5, 2, height_ratios=[1, 1, 1, 1, 1], width_ratios=[1, 1])
        ax0 = fig.add_subplot(grid[:, 0])
        ax2 = fig.add_subplot(grid[3:, -1], projection=rotatedpole)
        ax1 = fig.add_subplot(grid[0:2, -1], projection=rotatedpole)
        ax3 = fig.add_subplot(grid[2, -1])
        plt.subplots_adjust(hspace=0.12, wspace=0.05)

        # axes width match up with rotated pole projection
        # https://stackoverflow.com/questions/15480113/force-aspect-ratio-for-a-map
        ax1.set_adjustable('datalim')
        ax2.set_adjustable('datalim')

        extent0 = self.config["extent"]
        extent = setExtentByRatio(extent0, angle=90 - self.df["strike"].iloc[0] % 180)
        ax1.set_extent(extentZoom(extent, np.abs(rotatediff)), crs=ccrs.PlateCarree())
        ax2.set_extent(extentZoom(extent, np.abs(rotatediff)), crs=ccrs.PlateCarree())
        ax0pos = ax0.get_position().bounds
        ax1pos = ax1.get_position().bounds
        ax2pos = ax2.get_position().bounds

        if self.name == "":
            for x in [ax1, ax2]:
                x.add_wms(
                    # wms="https://www.gmrt.org/services/mapserver/wms_merc?",
                    wms="https://www.gebco.net/data_and_products/gebco_web_services/2019/mapserv?",
                    layers=["GEBCO_2019_Grid"],
                )
        else:
            def getBathymetryFromGMRTByExtent(name):
                f = netCDF4.Dataset(os.path.join(GMRT_BATHY_GRID, name + ".grd"))
                x_range = f.variables["x_range"][:]
                y_range = f.variables["y_range"][:]
                dims = f.variables["dimension"][:]
                xx = np.linspace(x_range[0], x_range[1], dims[0])
                yy = np.linspace(y_range[0], y_range[1], dims[1])
                zz = f.variables["z"][:]
                # zz = -np.abs(zz) # wrong GMRT data perhaps
                zz = np.reshape(zz, (dims[1], dims[0]))
                return xx, yy, zz, [x_range[0], x_range[1], y_range[0], y_range[1]]

            bathylon, bathylat, bathyelv, bathyextent = getBathymetryFromGMRTByExtent(name)
            maxelv, minelv = np.nanmin(bathyelv), np.nanmax(bathyelv)
            elvmaxnorm = maxelv + (minelv - maxelv) * 1.1
            dx = bathylon[len(bathylon)//2] - bathylon[len(bathylon)//2-1]
            dy = bathylat[len(bathylat)//2] - bathylat[len(bathylat)//2-1]
            dy = 111200 * dy
            dx = 111200 * dx * np.cos(np.radians(np.nanmean(bathylat))) # dx vary with latitude
            ls = LightSource(azdeg=self.df["strike"].values[0]-45.0, altdeg=45)
            cmapBathy = plt.cm.get_cmap("YlGnBu_r")
            origin = "upper"
            shade_data = np.where(np.isnan(bathyelv.data), 0.0, bathyelv.data)
            z = ls.shade(
                shade_data, # nan val corrupt lightsource
                cmap=cmapBathy, vert_exag=50.0, dx=dx, dy=dy, blend_mode="overlay", fraction=1.0,
                vmin=maxelv, vmax=elvmaxnorm)
            ax1.get_position().bounds
            ax2.get_position().bounds
            ax2.imshow(z, extent=bathyextent, origin=origin, transform=ccrs.PlateCarree(), cmap=cmapBathy, interpolation='nearest')
            ax1.imshow(z, extent=bathyextent, origin=origin, transform=ccrs.PlateCarree(), cmap=cmapBathy, interpolation='nearest')

        gl = ax1.gridlines(draw_labels=True, color='black', alpha=0.5, linestyle='--', zorder=1, x_inline=False, y_inline=False)
        gl.ylabels_left = False
        gl.ylabels_right = True
        gl.xlabels_top = True
        gl.xlabels_bottom = False
        gl.xlocator = mticker.MaxNLocator(4)
        gl.ylocator = mticker.MaxNLocator(4)

        gl = ax2.gridlines(draw_labels=True, color='black', alpha=0.5, linestyle='--', zorder=1, x_inline=False, y_inline=False)
        gl.ylabels_left = False
        gl.ylabels_right = True
        gl.xlabels_top = False
        gl.xlabels_bottom = True
        gl.xlocator = mticker.MaxNLocator(4)
        gl.ylocator = mticker.MaxNLocator(4)

        scale_bar(ax1, (0.80, 0.10), self.config["ruler"], color='black', angle=0, text_offset=0.015)
        xdist = dist_func_complete(ax1, [0.0, 0.5], [1.0, 0.5])
        ratio = self.config["ruler"] / xdist * 1e3
        ax0.plot([0.8, 0.8 + ratio], [datetime(1952, 1, 1), datetime(1952, 1, 1)], lw=3.0, color="k")
        ax0.text(0.8 + ratio / 2, datetime(1953, 1, 1), s=f"{self.config['ruler']:.0f} km" ,horizontalalignment='center', verticalalignment='center')

        ax0.set_ylim(bottom=clipped_date_post1950, top=datetime(2022, 1, 1))
        ax0.set_xlim(left=0, right=1)
        ax0.xaxis.set_major_formatter(NullFormatter())
        ax0.xaxis.set_ticks_position('none')
        ax0.set_xticklabels([])
        ax0.set_ylabel("Time")
        ax0.yaxis.set_major_locator(mdates.YearLocator(10))
        ax0.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax0.grid(True, which="major", axis="x")
        ax0.set_xlabel("Scaled Along Strike Position")
        ax0.tick_params(labelbottom=True, labeltop=True)

        ax3.set_xticklabels([])
        ax3.set_xlim(left=0, right=1)
        ax3.set_xticks([])
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position('right')
        ax3.set_xlabel("Scaled Along Strike Position", labelpad=2.0, fontsize="x-small")

        def full_extent(ax, pad=0.0):
            # https://stackoverflow.com/questions/14712665/matplotlib-subplot-background-axes-face-labels-colour-or-figure-axes-coor
            """Get the full extent of an axes, including axes labels, tick labels, and
            titles."""
            # For text objects, we need to draw the figure first, otherwise the extents
            # are undefined.
            ax.figure.canvas.draw()
            items = ax.get_xticklabels() + ax.get_yticklabels()
            items += [ax, ax.title]
            items += [ax.xaxis.get_label()]
            bbox = Bbox.union([item.get_window_extent() for item in items])
            return bbox.expanded(1.0 + pad, 1.0 + pad)

        # cax = ax1.inset_axes([0.635, 0.18, 0.3, 0.034])
        # cax.set_facecolor([1, 1, 1, 0.65])
        # cbar = fig.colorbar(im, cax=cax, orientation="horizontal", extend='both', shrink=0.9)
        # cax.patch.set_facecolor('black')
        # cbar.ax.set_xlabel('Water Depth (m)')
        # cbar.ax.xaxis.set_major_locator(mticker.LinearLocator(numticks=3))
        # extentBg = full_extent(cax)
        # extentBg = extentBg.transformed(ax1.transAxes.inverted())
        # rect = Rectangle([extentBg.xmin, extentBg.ymin], extentBg.width, extentBg.height,
        #     facecolor=[1.0, 1.0, 1.0, 0.65], edgecolor='none', zorder=3, # notice zorder is per-axis, not global
        #     transform=ax1.transAxes)
        # ax1.patches.append(rect)

        at = AnchoredText(f"{self.name} | {self.df['Vpl (mm/yr)'].iloc[0]:.0f} mm/yr | {self.df['Length (km)'].iloc[0]:.0f} km", loc='upper right', frameon=True,)
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax1.add_artist(at)

        def _get_ax_scale_ratio(xlim, ylim, xspan, yspan):
            return ((ylim[1] - ylim[0]) / yspan) / ((xlim[1] - xlim[0]) / xspan) * figsize[0] / figsize[1]

        ax0scaleratio = _get_ax_scale_ratio(ax0.get_xlim(), ax0.get_ylim(), ax0.get_position().bounds[2], ax0.get_position().bounds[3])
        ax1scaleratio = _get_ax_scale_ratio(ax1.get_xlim(), ax1.get_ylim(), ax1.get_position().bounds[2], ax1.get_position().bounds[3])
        beachballSizeFunc = lambda x: (x - 3.0) / 70 # notice here xlim [0, 1]
        beachballSizeOnMapFunc = lambda x: beachballSizeFunc(x) * (ax1.get_xlim()[1] - ax1.get_xlim()[0])
        # https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size/47403507#47403507
        # https://stackoverflow.com/questions/48172928/scale-matplotlib-pyplot-axes-scatter-markersize-by-x-scale/48174228#48174228
        sqrtsizeinpoint = ax0.get_window_extent().width / (ax0.get_xlim()[1] - ax0.get_xlim()[0]) * 72 / fig.dpi

        # scaling of scatter plot
        normalization = 12
        dftmp = dfMerged.copy()
        lats = dftmp["lat"].to_numpy() + np.array([self.config["adjust"].get(str(x), {'lat': 0, 'lon': 0})['lat'] for x in dftmp["group"]])
        lons = dftmp["lon"].to_numpy() + np.array([self.config["adjust"].get(str(x), {'lat': 0, 'lon': 0})['lon'] for x in dftmp["group"]])
        dftmp['lat'] = lats
        dftmp['lon'] = lons
        dftmp = dftmp[(dftmp['lat'] >= extent0[2]) & (dftmp['lat'] <= extent0[3]) & (dftmp['lon'] >= extent0[0]) & (dftmp['lon'] <= extent0[1])]
        dftmp['time'] = pd.to_datetime(dftmp['time'])
        dftmp = dftmp[dftmp['time'] > datetime(1990, 1, 1,)]
        dftmp['istransform'] = dftmp['id'].apply(lambda x: isTransform(self.fm[x]['np1']))
        dftmp = dftmp[dftmp['istransform']]
        if dftmp.shape[0] == 0:
            mmag = dfMerged["mag"].max()
        else:
            dftmp = dftmp[dftmp['group'] != -1]
            mmag = dftmp["mag"].max()
            if np.isnan(mmag):
                mmag = dfMerged["mag"].max()
        normalization *= (5.6 ** (mmag - 3)) / (5.6 ** (6.1 - 3))

        # beachballSizePt2x = lambda x: (x - 3.0) * 5 / sqrtsizeinpoint
        beachballSizePt2x = lambda x: 5.6 ** (x - 3.0) / normalization / sqrtsizeinpoint
        # beachballSizePt2x = lambda x: np.sqrt(10 ** (x * 3/2)) / 3e3 / sqrtsizeinpoint
        beachballSizePt2xOnMap = lambda x: beachballSizePt2x(x) * (ax1.get_xlim()[1] - ax1.get_xlim()[0])

        def _get_fm_fc_(eid, isrelocated=True):
            np1 = self.fm[str(eid)]["np1"]
            bcc = "tab:red" if isrelocated else "silver"
            if eid in anchors:
                bcc = "darkgoldenrod"
            if np.isnan(self.fm[str(eid)]["np1"][0]):
                if np.isnan(self.fm[str(eid)]["mt"][0]):
                    _fm = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0] # mimic scatter solid circle
                    bcc = "lightpink" if isrelocated else "white"
                else:
                    # _fm = fms[str(self.df.id.iloc[0])]["mt"] # not use, need rotation matrix
                    pass
            else:
                _fm = self.fm[str(eid)]["np1"]

            if len(_fm) != 6:
                _fm[0] += rotatediff
            return _fm, bcc

        def transform_t_day(t: str):
            _t = datetime.fromisoformat(t)
            _days = (_t - datetime(1970, 1, 1)).days + 1
            return _t, _days

        def transform_xy_xyproj(proj, x, y):
            xy = proj.transform_points(ccrs.PlateCarree(), np.array([x]), np.array([y]))[0]
            return xy[0], xy[1]

        def transform_xyproj_xyr(ax, xp, yp):
            return ax.transLimits.transform((xp, yp))

        def _add_beachball(r, isrelocated, fc="tab:red", forcedot=False):
            _t, _days = transform_t_day(r.time)
            _mag = r.mag
            if isrelocated:
                x, y = r.lon + self.config["adjust"][str(r.group)]["lon"], r.lat + self.config["adjust"][str(r.group)]["lat"]
                _zorder = 5
                _alpha = 1.0
            else:
                x, y = r.lon, r.lat
                _zorder = 2
                _alpha = 0.5
            xp, yp = transform_xy_xyproj(rotatedpole, x, y)
            xr, yr = transform_xyproj_xyr(ax1, xp, yp)
            if x < extent0[0] or x > extent0[1] or y < extent0[2] or y > extent0[3] or _t <= clipped_date_post1950:
                return
            widthbc0 = (beachballSizePt2x(_mag), beachballSizePt2x(_mag) * ax0scaleratio)
            widthbc1 = (beachballSizePt2xOnMap(_mag), beachballSizePt2xOnMap(_mag) * ax1scaleratio)
            _fm, bcc = _get_fm_fc_(r.id, isrelocated)
            bc0 = beach(_fm, alpha=_alpha, linewidth=1.0, facecolor=bcc, bgcolor="white", xy=(xr, _days), width=widthbc0, zorder=_zorder)
            ax0.add_collection(bc0)
            bc1 = beach(_fm, alpha=_alpha, linewidth=1.0, facecolor=bcc, bgcolor="white", xy=(xp, yp), width=widthbc1, zorder=_zorder)
            ax1.add_collection(bc1)
            if isrelocated:
                ax2.scatter(x, y, s=3**2, fc=bcc, ec="k", transform=ccrs.PlateCarree())

        adjust = self.config["adjust"]
        anchors = set()
        for k in adjust:
            if "anchor" in adjust[k]:
                rr = dfMerged[dfMerged["id"] == adjust[k]["anchor"]["id"]]
                anchors.add(adjust[k]["anchor"]["id"])
                adjust[k]["lon"] = adjust[k]["anchor"]["lon"] - rr.lon.values[0]
                adjust[k]["lat"] = adjust[k]["anchor"]["lat"] - rr.lat.values[0]
                print(f"adjust: {rr.group.values[0]}, lat: {adjust[k]['lat']}, lon: {adjust[k]['lon']}")

        for r in dfMerged.itertuples():
            _add_beachball(r, r.group != -1)

        attempt = datetime(1990, 1, 1) # cutoff of attempting relocation
        ax0.plot([0, 1], [attempt, attempt], linestyle="--", color="lightcoral")
        magcutoff = self.config["magCutOff"]
        ax0.text(0.02, datetime(1991, 6, 1), f"Relocate \nMw$\geq${magcutoff:.1f}", color="lightcoral", backgroundcolor=(0.9, 0.9, 0.9, 0.5), zorder=6)

        def momentHist(func=momentAlongStrikeEllipsoid):
            xs, mws = [], [] # all event after 1950
            xs2, mws2 = [], [] # all event after 1995/1990
            xs3, mws3 = [], [] # all relocated event
            for r in dfMerged.iloc[::-1].itertuples():
                fm = self.fm[str(r.id)]["np1"]
                if not isTransform(fm):
                    continue

                if r.group == -1:
                    x, y = r.lon, r.lat
                else:
                    x, y = r.lon + self.config["adjust"][str(r.group)]["lon"], r.lat + self.config["adjust"][str(r.group)]["lat"]

                if x < extent0[0] or x > extent0[1] or y < extent0[2] or y > extent0[3]:
                    continue
                    # pass

                xp, yp = transform_xy_xyproj(rotatedpole, x, y)
                xr, _ = transform_xyproj_xyr(ax1, xp, yp)
                xs.append(xr); mws.append(r.mag)
                if datetime.fromisoformat(r.time) >= clipped_date_post1990:
                    xs2.append(xr); mws2.append(r.mag)
                if r.group != -1:
                    xs3.append(xr); mws3.append(r.mag)

            xdist = dist_func_complete(ax1, [0.0, 0.5], [1.0, 0.5]) / 1000 # km
            arr, rr = func(xdist, xs, mws)
            arr2, rr2 = func(xdist, xs2, mws2)
            arr3, rr3 = func(xdist, xs3, mws3)
            ax3.plot(rr, arr / (2020-1950) / 1e13, color="cadetblue", label="since 1950")
            # ax3.plot(rr2, arr2 / (2020-1990) / 1e13, color="royalblue", label="since 1990")
            ax3.plot(rr3, arr3 / (2020-1990) / 1e13, color="deeppink", label="relocated only") # relocate attempt after 1990
            ax3.legend(
                loc="upper center", ncol=3, title="$E_{\mathrm{year}}[ \Sigma (M_{0}/L) ]$",
                # bbox_to_anchor=(0.5, 0.99),
                fontsize='xx-small', title_fontsize="xx-small")
            ax3.set_ylabel("Unit Moment Rate\n($ 10 ^{13} \cdot \mathrm{N} \;/ \;\mathrm{yr}$)")

            if "edge" in self.config:
                li = self.config["edge"]["l"]
                ri = self.config["edge"]["r"]
                edges = []
                for x in [li, ri]:
                    xp, yp = transform_xy_xyproj(rotatedpole, x[1], x[0])
                    xr, _ = transform_xyproj_xyr(ax1, xp, yp)
                    edges.append(xr)
                if edges[0] > edges[1]:
                    edges.reverse()

                # rect = Rectangle((edges[0], 0), edges[1] - edges[0], 1, transform=ax3.transAxes, fc="lightgray", alpha=0.6)
                # ax3.add_patch(rect)
                thrd = 0.1
                peak = np.max(arr3) / (2020-1990) / 1e13
                creepPct = creepPercentage(arr3 / (2020-1990), rr3, edges[0], edges[1], thrd)
                creepIndex, shape = creepMask(arr3 / (2020-1990), rr3, edges[0], edges[1], thrd)
                creepPct1950 = creepPercentage(arr / (2020-1950), rr, edges[0], edges[1], thrd)
                creepIndex1950, shape1950 = creepMask(arr / (2020-1950), rr, edges[0], edges[1], thrd)

                # plot 1950 or relocated
                rrp, cpidx = rr, creepIndex1950
                # rrp, cpidx = rr3, creepIndex
                for x in cpidx:
                    rect = Rectangle((rrp[x[0]], 0), rrp[x[1]] - rrp[x[0]], 1, transform=ax3.transAxes, fc="lightgray", alpha=0.6)
                    ax3.add_patch(rect)

                self.config["creepPercentage"] = creepPct
                self.config["creepPercentage1950"] = creepPct1950
                self.updateConfig()
                ax3.text(0.725, 0.9, f"CSF = {creepPct:.2f}|{creepPct1950:.2f}", transform=ax3.transAxes, ha="left", va="center", fontsize=9)

                expected = self.df["At"] * 1e6 * 3e10 * self.df["Vpl (mm/yr)"] / 1e3 / self.df["Length (km)"] / 1e3
                expected /= 1e13
                if not np.isnan(expected.values[0]):
                    ax3.plot([0.0, 1.0], [expected, expected], linestyle=":", color="forestgreen", linewidth=1.0)
                    ax3.text(0.9, expected, "$A_{T}$", color="forestgreen")

                with open(os.path.join(self.dir, "umrr.json"), "w") as fp:
                    d = {
                        "relocated": {
                            "x": list(rr3), "y": list(arr3 / (2020-1990) / 1e13),
                            "index": [int(y) for x in creepIndex for y in x],
                        },
                        "1950": {
                            "x": list(rr), "y": list(arr / (2020-1950) / 1e13),
                        },
                        "At": expected.values[0],
                    }
                    json.dump(d, fp, indent=4)
            ax3.set_ylim(bottom=0)

        momentHist()

        ls = []
        if mmag < 5.8:
            magLengendSize = [5.0, 5.3, 5.6]
        elif mmag < 6.3:
            magLengendSize = [5.0, 5.5, 6.0]
        elif mmag < 6.8:
            magLengendSize = [5.5, 6.0, 6.5]
        else:
            magLengendSize = [6.0, 6.5, 7.0]

        magLengendSize = [mmag - 1, mmag - 0.5, mmag]
        magLengendLabel = [f"{x:.1f}" for x in magLengendSize]
        for i in range(len(magLengendSize)):
            l = ax0.scatter([],[], ec="gray", fc="white", s=(beachballSizePt2x(magLengendSize[i])*sqrtsizeinpoint)**2)
            ls.append(l)
        leg = ax0.legend(
            ls, magLengendLabel, ncol=3, frameon=True, borderpad=0.4, bbox_to_anchor=(0.5, 1.0),
            loc="lower center", title="Mw", fancybox=True, labelspacing=1.0, framealpha=1.0,
            handletextpad=1.2, borderaxespad=1.2, fontsize='small', title_fontsize="small")

        for t, x in zip(["A", "B", "C", "D"], [ax0, ax1, ax3, ax2]):
            loc = "upper left"
            at = AnchoredText(t, loc=loc, frameon=True,)
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            x.add_artist(at)

        if self.name == "Gofar":
            ax2.text(0.3, 0.55, "G3", transform=ax2.transAxes, color="Purple", backgroundcolor="white")
            ax2.text(0.56, 0.62, "G2", transform=ax2.transAxes, color="Purple", backgroundcolor="white")
            ax2.text(0.76, 0.70, "G1", transform=ax2.transAxes, color="Purple", backgroundcolor="white")

        sub_ax = fig.add_axes([ax2pos[0]-0.01, ax2pos[1], 0.10, 0.10], projection=ccrs.NearsidePerspective(self.df["Longitude"].iloc[0], self.df["Latitude"].iloc[0]))
        sub_ax.set_adjustable("datalim")
        sub_ax.stock_img()
        sub_ax.scatter(self.df["Longitude"].iloc[0], self.df["Latitude"].iloc[0], s=50, marker="*", c="orange")

        fig.savefig(output, dpi=600, bbox_inches="tight")
        plt.close(fig)

def plotting(name):
    f = PlotProcedure(name.strip())
    f.plotTimeSpaceLayout()

if __name__ == "__main__":

    names = dfFaults["Name"].to_list()
    for i, name in enumerate(names):
        print(f"{i + 1}/{len(names)}: {name}")
        f = PlotProcedure(name.strip())
        f.plotTimeSpaceLayout()

    # parallel plotting does not correctly render figures
    # with ProcessPoolExecutor(max_workers=2) as executor:
    #     futures = []
    #     for i, name in enumerate(names):
    #         print(f"{i + 1}/{len(names)}: {name}")
    #         futures.append(executor.submit(plotting, name))
    #     wait(futures)
