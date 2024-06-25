# CommunityGrowthPrediction
An Independent Study Project conducted as part of the ACENET Microcredential in Advanced Computing.

This project is an attempt to predict population growth over time using a machine learning model, given a number of population characteristics from one or more census years, and geospatial information related to community layout, particularly road network characteristics. Specifically, this project is an attempt to answer the following research questions:

RQ 1: Can census data from several different census years be used to train a model to
predict population change over time within a limited geographical scope (eg. provincial
scale)?

RQ 2: Does the inclusion of geospatial characteristics ( road network characteristics,
building density, population density, etc.) include the prediction capability of the model
from RQ1?

RQ 3: Do prediction capabilities persist when geographical scope is expanded to
several provinces, or to a Canada-wide analysis?

To conduct this analysis, two primary datasets will be used. 

Census data from [Statistics Canada](https://www.statcan.gc.ca/en/start), accessed primarily using a python library called [stats_can](https://github.com/ianepreston/stats_can) that interfaces with the [Statistics Canada API](https://www.statcan.gc.ca/en/developers/wds) will be used to source aggregate population and demographic data. If each community we use for training consitutes a row, there are hundreds of columns we can include programatically to serve as parameters for the machine learning model. Depending on the performance of the python library mentioned above, this project may use the entirety of the StatCan dataset as the parameter space for community characteristics, or columns may be specifically constructed to focus on population change, age characteristics, household characteristics, and details related to occupation and employment. There is also an oppotunity to calculate a variety of sub-characteristics, like **Population and Job Balance**, which measures the ratio between population and jobs within each community, although use of these characteristics will be a stretch goal. 

Geospatial data will be taken from [OpenStreetMap]((https://www.openstreetmap.org/#map=15/47.5580/-52.7050&layers=H)). The data of interest will be building data, accessed using the well-established python library [OSMPythonTools](https://wiki.openstreetmap.org/wiki/OSMPythonTools), and road network data, which will be collected using [OSMnx](https://osmnx.readthedocs.io/en/stable/), a python package built on top of [NetworkX](https://networkx.org/) for detailed network analysis on road networks taken from OpenStreetMap. Building data in particular is often sparse for smaller communities, so if preprocessing shows significant gaps in the data, this part of the project may focus more on road network characteristics, which is typically a more reliable subset of OpenStreetMap data. Key building characteristics are building density, land use diversity index, residential land-use mix, and urban forest, which will all need to be extracted from the building data during data preprocessing. Key road network characteristics selected here are number that should be directly or indirectly relevant to travel along the road network. They are [node connectivity](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.connectivity.node_connectivity.html#networkx.algorithms.approximation.connectivity.node_connectivity), [betweenness centrality](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html#networkx.algorithms.centrality.betweenness_centrality), and the [average neighbor degree](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.assortativity.average_neighbor_degree.html#networkx.algorithms.assortativity.average_neighbor_degree). All of these can be readily calculated using networkx. 

A similar study on predicting population growth can be found [here](https://www.mecs-press.org/ijeme/ijeme-v13-n2/IJEME-V13-N2-1.pdf)
