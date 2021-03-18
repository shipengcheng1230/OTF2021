import httpx
import asyncio
import aiofiles
import os
import json

from pyproj import Geod
from utils import *


async def download(client, name, extent, dextent=1.0):
    print(f"Start download {name}.")
    async with client.stream("GET", "", params={
        'minlongitude': extent[0] - dextent,
        'maxlongitude': extent[1] + dextent,
        'minlatitude': extent[2] - dextent,
        'maxlatitude': extent[3] + dextent,
        }) as st:
        outfile = os.path.join(os.path.dirname(__file__), "..", "GMRT", name + ".grd")
        if os.path.isfile(outfile) and os.path.getsize(outfile) > 10*1024:
            return name
        async with aiofiles.open(os.path.join(outfile), "wb") as f:
            async for data in st.aiter_bytes():
                await f.write(data)
    print(f"Finsh download {name}.")
    return name


async def main():
    async with httpx.AsyncClient(
        base_url="https://www.gmrt.org/services/GridServer?", # GMRT multiresolution data
        params={
            'format': 'netcdf',
            'resolution': 'max',
            'layer': 'topo', # masked with GEBCO 2014
            },
        timeout=httpx.Timeout(read_timeout=None),
        ) as client:
        ts = []
        for r in dffault.itertuples():
            with open(os.path.join(os.path.dirname(__file__), "..", "catalogue-adjust", r.Name + ". json"), "r") as fp:
                extent = json.load(fp)["extent"]
            t = asyncio.create_task(download(client, r.Name, extent))
            ts.append(t)
        await asyncio.gather(*ts)

def download_seq(client, name, extent, dextent=1.0):
     outfile = os.path.join(os.path.dirname(__file__), "..", "bathy", name + ".grd")
     if os.path.isfile(outfile) and os.path.getsize(outfile) > 10*1024:
         print(f"Skip download {name}.")
         return name

     print(f"Start download {name}.")
     with client.stream("GET", "", params={
         'minlongitude': extent[0] - dextent,
         'maxlongitude': extent[1] + dextent,
         'minlatitude': extent[2] - dextent,
         'maxlatitude': extent[3] + dextent,
         }) as st:
         with open(outfile, "wb") as f:
             for data in st.iter_raw():
                 f.write(data)
     print(f"Finsh download {name}.")
     return name


def main_seq():

    with httpx.Client(
        base_url="https://www.gmrt.org/services/GridServer?", # GMRT multiresolution data
        params={
            'format': 'netcdf',
            'resolution': 'max',
            'layer': 'topo', # masked with GEBCO 2014
            },
        timeout=None,
    ) as client:

        for r in dfFaults.itertuples(index=True):
            print(f"{r.Index}")
            if isinstance(r.Name, float):
                break
            with open(os.path.join(os.path.dirname(__file__), "..", "faults", r.Name, "config.json"), "r") as fp:
                extent0 = json.load(fp)["extent"]
                extent = setExtentByRatio(extent0, angle=90 - r.strike % 180)
                # print(extent)
                download_seq(client, r.Name, extent)

if __name__ == "__main__":
    while True:
        try:
            main_seq()
        except:
            continue
        finally:
            break
