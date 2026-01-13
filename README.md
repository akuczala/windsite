# WindSite

WindSite is an application that identifies ideal locations for future wind farm development. Building on work done by NREL, this application narrows the search for wind farm sites by leveraging multiple data sources and machine learning.

## Instructions

The app balances 3 weighted factors in order to recommend sites
- Transmission line proximity
- Road proximity
- Estimated price per acre

The user can adjust the relative importance of these factors by adjusting the three weight sliders.

The app also allows users to set constraints to filter potential sites:
- Set the maximum allowable distance to a transmission line (0-50 miles)
- Set the maximum allowable ditance to a road (0-6 miles)
- Set the minimum distance to a residential area (0-6 miles)
