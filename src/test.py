from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException

c =  Client("USGS")
c.get_events(eventid=eid)