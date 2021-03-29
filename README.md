# Oceanic Transform Faults

This repo consists of relocated (surface wave cross-correlation) earthquakes in major oceanic transform faults.

*The ubiquitous creeping segments on oceanic transform faults, Pengcheng Shi, Meng Wei, Robert Pockalny, 2021, to submit to Geology*

### What includes:

- `/src`: code to reproduce this work.

- `/faults`: data and intermediate products on each OTF

    - We use UGSG *event_id*, so you can view a event information by visiting `https://earthquake.usgs.gov/earthquakes/eventpage/{event_id}`

    - `mt.json`: moment tensor solutions (the latest one) from USGS server

    - `config.json`: plotting parameters and CSF results

    - `stations.csv`: stations we tried to retrieve waveform

    - `umrr.json`: arrays of the unit moment release rate


- `/imgs`: spatial-temporal seismic activites on each OTF

- `faults.xlsx`: the table of OTF parameters and results

- `stations.csv`: our selected seismic station pool

- `extra`:

    - `ISC-EHB`: the corresponding catalogs from ISC-EHB for Fig-1

    - `cosine-fitting-samples`: intermediate results from from SEIR 96E (B)


### What excludes:

- seismic waveforms

- complete cosine fitting results
