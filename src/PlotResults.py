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

class PlotProcedure(AbstractFaultProcess):

    def __init__(self, name):
        super().__init__(name)
        with open(os.path.join(self.dir, "mt.json")) as fp:
            self.fm = json.load(fp)

    def plotTimeSpaceLayout2(self, overWrite=True):
        output = os.path.join(self.dir, "layout2.pdf")
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

        def extentZoom(extent, angle):
            llx, urx, lly, ury = extent
            midx, midy = (llx + urx) / 2, (lly + ury) / 2
            dx, dy = urx - llx, ury - lly
            zoom = (0.5 - 0.95) * (1 - np.abs(angle - 45) / 45) + 0.95 # linear expand
            _dx, _dy = dx * zoom / 2, dy * zoom / 2
            return [midx - _dx, midx + _dx, midy - _dy, midy + _dy]

        def setExtentByRatio(extent, ratio=2.00, angle=None):
            geod = Geod(ellps="WGS84")
            x0 = angularMean([extent[0], extent[1]])
            y0 = angularMean([extent[2], extent[3]])
            dx = extent[1] - extent[0]
            dy = extent[3] - extent[2]
            _, _, ddx = geod.inv(x0, y0, x0+dx/2, y0)
            _, _, ddy = geod.inv(x0, y0, x0, y0+dy/2)
            _d1, _d2 = max(ddx, ddy), min(ddx, ddy)
            if _d1 / _d2 > ratio:
                _d2 = _d1 / ratio
            else:
                _d1 = _d2 * ratio
            if angle:
                if np.abs(angle) < 45:
                    ddx, ddy = _d1, _d2
                else:
                    ddx, ddy = _d2, _d1
            else:
                if dx > dy:
                    ddx, ddy = _d1, _d2
                else:
                    ddx, ddy = _d2, _d1
            maxlon, _, _ = geod.fwd(x0, y0, 90.0, ddx)
            minlon, _, _ = geod.fwd(x0, y0, -90.0, ddx)
            _, maxlat, _ = geod.fwd(x0, y0, 0.0, ddy)
            _, minlat, _ = geod.fwd(x0, y0, 180.0, ddy)
            return [minlon, maxlon, minlat, maxlat]

        extent0 = self.config["extent"]
        extent = setExtentByRatio(extent0, angle=90 - self.df["strike"].iloc[0] % 180)
        ax1.set_extent(extentZoom(extent, np.abs(rotatediff)), crs=ccrs.PlateCarree())
        ax2.set_extent(extentZoom(extent, np.abs(rotatediff)), crs=ccrs.PlateCarree())
        ax0pos = ax0.get_position().bounds
        ax1pos = ax1.get_position().bounds
        ax2pos = ax2.get_position().bounds

        for x in [ax1, ax2]:
            x.add_wms(
                wms="https://www.gmrt.org/services/mapserver/wms_merc?",
                layers=["GMRT"],
                cmap=plt.cm.get_cmap("ocean"),
            )
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

        scale_bar(ax1, (0.80, 0.90), self.config["ruler"], color='black', angle=0, text_offset=0.015)
        xdist = dist_func_complete(ax1, [0.0, 0.5], [1.0, 0.5])
        ratio = self.config["ruler"] / xdist * 1e3
        ax0.plot([0.1, 0.1 + ratio], [datetime(1952, 1, 1), datetime(1952, 1, 1)], lw=3.0, color="k")
        ax0.text(0.1 + ratio / 2, datetime(1953, 1, 1), s=f"{self.config['ruler']:.0f} km" ,horizontalalignment='center', verticalalignment='center')

        ax0.set_ylim(bottom=clipped_date_post1950, top=datetime(2022, 1, 1))
        ax0.set_xlim(left=0, right=1)
        ax0.xaxis.set_major_formatter(NullFormatter())
        ax0.xaxis.set_ticks_position('none')
        ax0.set_xticklabels([])
        ax0.set_ylabel("Occurrence Time")
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

        sub_ax = fig.add_axes([ax2pos[0]-0.05, ax2pos[1], 0.10, 0.10], projection=ccrs.NearsidePerspective(self.df["Longitude"].iloc[0], self.df["Latitude"].iloc[0]))
        sub_ax.stock_img()
        sub_ax.scatter(self.df["Longitude"].iloc[0], self.df["Latitude"].iloc[0], s=50, marker="*", c="orange")

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

        at = AnchoredText(f"{self.name} | {self.df['Vpl (mm/yr)'].iloc[0]:.0f} mm/yr | {self.df['Length (km)'].iloc[0]:.0f} km", loc='upper left', frameon=True,)
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
        beachballSizePt2x = lambda x: (x - 3.0) * 5 / sqrtsizeinpoint
        beachballSizePt2xOnMap = lambda x: beachballSizePt2x(x) * (ax1.get_xlim()[1] - ax1.get_xlim()[0])

        def _get_fm_fc_(eid, isrelocated=True):
            np1 = self.fm[str(eid)]["np1"]
            bcc = "tab:red" if isrelocated else "silver"
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

        dfMerged = pd.read_csv(os.path.join(self.dir, "catalog-merged.csv"))
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
            ax3.plot(rr, arr / (2020-1950) / 1e20, color="lightslategray", label="since 1950")
            ax3.plot(rr2, arr2 / (2020-1990) / 1e20, color="royalblue", label="since 1990")
            ax3.plot(rr3, arr3 / (2020-1990) / 1e20, color="deeppink", label="relocated only") # relocate attempt after 1990
            ax3.legend(
                loc="lower center", ncol=3, title="$E_{\mathrm{year}}[ \Sigma (M_{0}/L) ]$",
                bbox_to_anchor=(0.5, 0.80), fontsize='xx-small', title_fontsize="xx-small")
            ax3.set_ylim(bottom=0)
            ax3.set_ylabel("Unit Moment Rate\n($ 10 ^{20} \cdot \mathrm{N} \;/ \;\mathrm{yr}$)")

            if "edge" in self.config:
                l = self.config["edge"]["l"]
                r = self.config["edge"]["r"]
                edges = []
                for x in [l, r]:
                    xp, yp = transform_xy_xyproj(rotatedpole, x[1], x[0])
                    xr, _ = transform_xyproj_xyr(ax1, xp, yp)
                    edges.append(xr)

                rect = Rectangle((edges[0], 0), edges[1] - edges[0], 1, transform=ax3.transAxes, fc="lightgray", alpha=0.6)
                ax3.add_patch(rect)
                creepPct = creepPercentage(arr3 / (2020-1990), rr3, edges[0], edges[1], 0.05)
                self.config["creepPercentage"] = creepPct
                self.updateConfig()

        momentHist()

        ls = []
        magLengendSize = [5.0, 5.5, 6.0]
        magLengendLabel = ["5.0", "5.5", "6.0"]
        for i in range(len(magLengendSize)):
            l = ax0.scatter([],[], ec="gray", fc="white", s=(beachballSizePt2x(magLengendSize[i])*sqrtsizeinpoint)**2)
            ls.append(l)
        leg = ax0.legend(
            ls, magLengendLabel, ncol=3, frameon=True, borderpad=0.4, bbox_to_anchor=(0.5, 1.0),
            loc="lower center", title="Mw", fancybox=True, labelspacing=1.0, framealpha=1.0,
            handletextpad=0.2, borderaxespad=1.2, fontsize='small', title_fontsize="small")

        print(f"Saving link simple plot {self.name} ...")
        fig.savefig(output, dpi=600)
        plt.close(fig)

if __name__ == "__main__":

    df2 = dfFaults.loc[(dfFaults["Good Bathymetry"] == 1) & (dfFaults["key"] > 79)]
    names = df2["Name"].to_list()

    for name in ["Discovery"]:
        f = PlotProcedure(name.strip())
        f.plotTimeSpaceLayout2()
