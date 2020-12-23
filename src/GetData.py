import datetime
import heapq
import json
import os
from collections import namedtuple
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed, wait)
from typing import *
from urllib.parse import parse_qs, urlparse

import cartopy.crs as ccrs
import geopy
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
from geopy import distance
from obspy import UTCDateTime
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException
from pyproj import Geod

from utils import *


class AbstractFaultProcess:

    def __init__(self, name):
        self.name = name
        self.df = dfFaults.loc[dfFaults['Name'] == self.name]
        self.dir = os.path.join(faultsDir, name)
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        if not os.path.isfile(config := os.path.join(self.dir, "config.json")):
            with open(config, "w") as fp:
                json.dump(
                    {
                        "ref": {"lat": 0.0, "lon": 0.0},
                        "magCutOff": 5.0,
                        "adjust": {
                            "1": {"lon": 0.0, "lat": 0.0}
                        },
                        "extent": [-180, 180, -90, 90],
                        "ruler": 10,
                    },
                    fp, indent=4)
        with open(config, "r") as fp:
            self.config = json.load(fp)


    def updateConfig(self):
        with open(os.path.join(self.dir, "config.json"), "w") as fp:
            json.dump(self.config, fp, indent=4)


class FaultData(AbstractFaultProcess):

    def __init__(self, name, catalogClient="USGS", waveClient="IRIS"):
        super().__init__(name)
        self.clientUSGS = Client(catalogClient, timeout=30)
        self.clientIRIS = Client(waveClient, timeout=30)

    def getCatalog(self, minMag=5.0, extent=0.5,
                   startTime=UTCDateTime("1950-01-01"),
                   endTime=UTCDateTime("2020-12-01"),
                   suffix="",
                   mt=True,
                   ):

        try:
            cat = self.clientUSGS.get_events(
                starttime=startTime,
                endtime=endTime,
                minmagnitude=minMag,
                minlatitude=self.df['minlat'].iloc[0] - extent,
                maxlatitude=self.df['maxlat'].iloc[0] + extent,
                minlongitude=self.df['minlon'].iloc[0] - extent,
                maxlongitude=self.df['maxlon'].iloc[0] + extent,
                orderby="time",
            )
        except FDSNNoDataException:
            print(f"No data for {self.faultName}!")
            cat = []

        content = {x: [] for x in ["time", "mag",
                                   "magType", "lat", "lon", "dep", "text", "id"]}
        content["time"] = [str(x.origins[0].time.datetime) for x in cat]
        content["mag"] = [x.magnitudes[0].mag for x in cat]
        content["magType"] = [x.magnitudes[0].magnitude_type for x in cat]
        content["lat"] = [x.origins[0].latitude for x in cat]
        content["lon"] = [x.origins[0].longitude for x in cat]
        content["dep"] = [x.origins[0].depth for x in cat]
        content["text"] = [x.event_descriptions[0].text for x in cat]
        content["id"] = [parse_qs(urlparse(x.resource_id.id).query)[
            "eventid"][0] for x in cat]
        pd.DataFrame(content).to_csv(os.path.join(
            self.dir, "catalog" + suffix + ".csv"), index=False)

        if not mt:
            return

        mt = {}
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures2key = {executor.submit(
                FaultData.getMomentTensorByEventID, self.clientUSGS, eid): eid for eid in content["id"]}
            for future in as_completed(futures2key):
                eid = futures2key[future]
                try:
                    data = future.result()
                except Exception as e:
                    raise(e)
                else:
                    mt[eid] = {"np1": data[0], "np2": data[1], "mt": data[2]}
            with open(os.path.join(self.dir, "mt.json"), "w") as fp:
                json.dump(mt, fp, indent=4)

    @staticmethod
    def getMomentTensorByEventID(client, eid):
        e = client.get_events(eventid=eid)
        fm = e[0].focal_mechanisms
        if not fm:
            return [np.nan for _ in range(3)], [np.nan for _ in range(3)], [np.nan for _ in range(6)]
        for _fm in reversed(fm):  # the later the better
            preferred_fm = _fm
            if _fm.nodal_planes or _fm.moment_tensor.tensor:
                break
        ts = preferred_fm.moment_tensor.tensor
        npp = preferred_fm.nodal_planes
        if not ts:
            mt = [np.nan for _ in range(6)]
        else:
            mt = [ts.m_rr, ts.m_tt, ts.m_pp, ts.m_rt, ts.m_rp, ts.m_tp]
        if not npp:
            np1, np2 = [np.nan for _ in range(3)], [np.nan for _ in range(3)]
        else:
            np1 = [npp.nodal_plane_1.strike,
                   npp.nodal_plane_1.dip, npp.nodal_plane_1.rake]
            np2 = [npp.nodal_plane_2.strike,
                   npp.nodal_plane_2.dip, npp.nodal_plane_2.rake]
        return np1, np2, mt

    def getCandidateStations(self, minDist=1000.0, ddeg=6.0, nnearest=1, plot=True):
        NET = dfStations['#Network '].to_list()
        STA = dfStations[' Station '].to_list()
        LAT = dfStations[' Latitude '].to_list()
        LON = dfStations[' Longitude '].to_list()
        DEP = dfStations[' Elevation '].to_list()

        degs = np.arange(0.0, 359.0, step=ddeg)
        sep = [(degs[x], degs[x+1]) for x in range(len(degs)-1)]
        sep.append((degs[-1], 360.0))

        geod = Geod(ellps="WGS84")
        staType = namedtuple(
            'station', ['net', 'sta', 'lat', 'lon', 'dep', 'azi', 'dist'])
        stas = [[] for _ in range(len(sep))]
        lon, lat = self.df["Longitude"].iloc[0], self.df["Latitude"].iloc[0]
        for i in range(len(STA)):
            fz, _, d = geod.inv(lon, lat, LON[i], LAT[i])
            if fz < 0:
                fz += 360
            d /= 1e3
            if d < minDist:
                continue
            id = int(np.floor(fz / ddeg))
            stas[id].append(
                staType(NET[i], STA[i], LAT[i], LON[i], DEP[i], fz, d))

        stass = []
        for sta in stas:
            heapq.heapify(sta)
            stass.extend(heapq.nsmallest(nnearest, sta, key=lambda x: x.dist))

        content = {
            'net': [x.net for x in stass],
            'sta': [x.sta for x in stass],
            'lat': [x.lat for x in stass],
            'lon': [x.lon for x in stass],
            'dep': [x.dep for x in stass],
            'azi': [x.azi for x in stass],
            'dist': [x.dist for x in stass],
        }
        pd.DataFrame(content).to_csv(os.path.join(
            self.dir, "stations.csv"), index=False)

        if not plot:
            return

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(lon))
        ax.stock_img()
        ax.coastlines()
        for i in range(len(content['sta'])):
            ax.plot([content['lon'][i], lon], [content['lat'][i], lat],
                    transform=ccrs.Geodetic(), linestyle=':', color="tab:red", zorder=1)
        ax.scatter(lon, lat, s=20, marker="*", c="orange",
                   transform=ccrs.PlateCarree(), zorder=2)
        ax.scatter(content['lon'], content['lat'], s=10, marker='s',
                   c='tab:purple', transform=ccrs.PlateCarree(), zorder=2)
        ax.set_title(self.name)
        fig.savefig(os.path.join(self.dir, "stations.pdf"))
        plt.close(fig)

    def getCandidateEvents(self,
                           timeFunc=lambda t: t >= datetime.datetime(
                               1990, 1, 1),
                           magFunc=lambda m1, m2, dm: m1 >= (m2 - dm)
                           ):
        df = pd.read_csv(os.path.join(self.dir, "catalog.csv"))
        refLat, refLon = self.config["ref"]["lat"], self.config["ref"]["lon"]
        magCutOff = self.config["magCutOff"]
        p0 = geopy.Point(refLat, refLon)
        content = {x: [] for x in [
            "id", "time", "lat", "lon", "mag", "along_strike_distance"
        ]}
        for r in df.itertuples(index=True):
            mw = mag2mw(r.mag, r.magType)
            if (
                timeFunc(datetime.datetime.fromisoformat(r.time)) and
                magFunc(mw, magCutOff, dm=0)
            ):
                content['id'].append(r.id)
                content['time'].append(r.time)
                content['lat'].append(r.lat)
                content['lon'].append(r.lon)
                content['mag'].append(mw)
                content['along_strike_distance'].append(
                    distance.distance(p0, geopy.Point(r.lat, r.lon)).km)
        pd.DataFrame(content).to_csv(os.path.join(
            self.dir, "catalog-candidate.csv"), index=False)

    @staticmethod
    def isPair(d1, m1, d2, m2, dc=125.0, mc=0.41, LARGE_MAG_PAIR=5.99):
        # notice, for instance, 5.4 - 5.2 > 0.2 is True
        return np.abs(d1 - d2) <= dc and \
            ((np.abs(m1 - m2) <= mc) or (m1 >= LARGE_MAG_PAIR and m2 >= LARGE_MAG_PAIR))

    def getEventPairs(self, *args, **kwargs):
        # denormalized table
        content = {x: [] for x in [
            "id1", "t1", "lat1", "lon1", "mag1", "dist1",
            "id2", "t2", "lat2", "lon2", "mag2", "dist2",
            "dmag", "ddist"]}
        df = pd.read_csv(os.path.join(self.dir, "catalog-candidate.csv"))
        for i in range(numrec := len(df.index)):
            r1 = df.loc[i]
            for j in range(i+1, numrec):
                r2 = df.loc[j]
                if FaultData.isPair(r1.along_strike_distance, r1.mag, r2.along_strike_distance, r2.mag, *args, **kwargs):
                    content['id1'].append(r1.id)
                    content['t1'].append(r1.time)
                    content['lat1'].append(r1.lat)
                    content['lon1'].append(r1.lon)
                    content['mag1'].append(r1.mag)
                    content['dist1'].append(r1.along_strike_distance)
                    content['id2'].append(r2.id)
                    content['t2'].append(r2.time)
                    content['lat2'].append(r2.lat)
                    content['lon2'].append(r2.lon)
                    content['mag2'].append(r2.mag)
                    content['dist2'].append(r2.along_strike_distance)
                    content['dmag'].append(np.abs(r1.mag - r2.mag))
                    content['ddist'].append(
                        np.abs(r1.along_strike_distance - r2.along_strike_distance))
        pd.DataFrame(content).to_csv(
            os.path.join(self.dir, "catalog-pair.csv"), index=False)

    def getWaveform(self,
                    groupVelocityWindow=[5.0, 3.0],  # km/s
                    filterHzWindow=[0.02, 0.04],
                    channel="BHZ",
                    ):
        waveDir = os.path.join(self.dir, "waves")
        if not os.path.isdir(waveDir):
            os.mkdir(waveDir)

        dfPairs = pd.read_csv(os.path.join(self.dir, "catalog-pair.csv"))
        dfStations = pd.read_csv(os.path.join(self.dir, "stations.csv"))

        def getWaveFormByVelocityWindow(tstr: str, net: str, sta: str, dist: float):
            dt1, dt2 = dist / \
                groupVelocityWindow[0], dist / \
                groupVelocityWindow[1]  # seconds
            t0 = UTCDateTime(tstr)
            st = self.clientIRIS.get_waveforms(
                net, sta, "*", channel, t0+dt1, t0+dt2, attach_response=True)
            try:
                st.remove_response()
            except:
                pass  # then no instrument removal performed
            finally:
                return st

        def getAndSave(eid, tstr, net, sta, dist):
            waveFileName = str(eid) + '-' + net + '-' + sta + '.sac'
            waveFilePath = os.path.join(waveDir, waveFileName)
            if os.path.isfile(waveFilePath) and os.path.getsize(waveFilePath) > 0:
                return
            try:
                with open(waveFilePath, "w"):
                    pass
                st = getWaveFormByVelocityWindow(tstr, net, sta, dist)
            except FDSNNoDataException:
                pass
            except Exception as e:
                pass
            else:
                if st and st.count != 0:
                    st.detrend(type="constant")
                    st.filter(
                        type="bandpass", freqmin=filterHzWindow[0], freqmax=filterHzWindow[1], zerophase=True)
                    print(f"Saving {self.name} - {waveFileName}")
                    st[0].write(waveFilePath, format="SAC")
            finally:
                if os.path.isfile(waveFilePath) and os.path.getsize(waveFilePath) == 0:
                    os.remove(waveFilePath)

        toDownload = \
            set(list(dfPairs[["id1", "t1"]].itertuples(index=False, name=None))) | \
            set(list(dfPairs[["id2", "t2"]].itertuples(
                index=False, name=None)))
        existed = [x for x in os.listdir(waveDir) if x.endswith(".sac")]

        for x in toDownload:
            for r in dfStations.itertuples():
                if not np.isnan(r.azi):
                    key = f"{x[0]}-{r.net}-{r.sta}.sac"
                    if (not existed) or key not in existed:
                        # obspy async client is not available, so bear this sequential download
                        # print(f"Trying {self.name}: {key} ...")
                        try:
                            getAndSave(x[0], x[1], r.net, r.sta, r.dist)
                        except:
                            continue
        return None


def getAllData(name: str):
    print(name)
    f = FaultData(name)
    f.getCatalog()
    # f.getCandidateStations()
    f.getCandidateEvents()
    f.getEventPairs()
    # f.getWaveform()
    return 0

if __name__ == "__main__":

    df2 = dfFaults.loc[(dfFaults["Good Bathymetry"] == 1) | (dfFaults["key"] < 80)]
    names = df2["Name"].to_list()
    for name in names:
        getAllData(name.strip())

    # You may get refused by the server if you open too many clients at the same time
    # You may not get all the waveform on the first request
    # Keep trying until no more
    # i = 0
    # while True:
    #     with ProcessPoolExecutor(max_workers=12) as executor:
    #         futures = []
    #         for name in names:
    #             futures.append(executor.submit(getAllData, name))
    #         try:
    #             # [future.result() for future in as_completed(futures)]
    #             wait(futures)
    #         except Exception as e:
    #             print(f"Retry {i} ......")
    #             i += 1
    #         else:
    #             # exit()
    #             pass
