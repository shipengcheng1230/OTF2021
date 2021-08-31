# Oceanic Transform Faults

This repo consists of relocated (surface wave cross-correlation) earthquakes in major oceanic transform faults.

*The ubiquitous creeping segments on oceanic transform faults, Pengcheng Shi, Meng Wei, Robert Pockalny, 2021, Geology (accepted)*

### What includes:

- `/src`: code to reproduce this work.

- `/faults`: data and intermediate products on each OTF

    - We use UGSG *event_id*, so you can view a event information by visiting `https://earthquake.usgs.gov/earthquakes/eventpage/{event_id}`

    - `mt.json`: moment tensor solutions (the latest one) from USGS server

    - `config.json`: plotting-related parameters, such as map extent, cluster offset, and CSF results

    - `stations.csv`: stations we tried to retrieve waveform

    - `umrr.json`: arrays of the unit moment release rate


- `/imgs`: spatial-temporal seismic activites on each OTF

- `faults.xlsx`: the table of OTF parameters and results

- `stations.csv`: our selected seismic station pool

- `/extra`:

    - `/ISC-EHB`: the corresponding catalogs from ISC-EHB for Fig-1

    - `/cosine-fitting-samples`: results from SEIR 96E (B)


### What excludes:

- seismic waveforms

- complete cosine fitting results


### Changelog

v1.0.1

In panel (C) of all supplementary figures of those earthquake relocation results, the green symbol $A_T$ is changed to $\dot{M}_{E}$, corresponding to the definition in the main context.

v1.0.0

Initial version.
