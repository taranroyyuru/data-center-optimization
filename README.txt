Hey Judges! This is Lev, here is a zip file with all of the optimization and sustainability scoring methods.
The data sets are too big to upload to github and we ran out of time to create an API to link to them.
The way the algorithm works is through creating a normalized score out of metrics we select from the main data csv,
which differ between linear, log, and exonential functinals that scale each metric properly.
The normalized metrics are obtained through analyzing the entire set, and defining bounds using the bottom 5 percentile
and the top 95 percentile. 

The multi stage optimization is a grid-like approach, where a grid is put over the US map, each location has the
sustainability index calculated, the highest indexes are kept, identifying hotspots, and the distance betwen grid points
and the radius of hotspots is lowered at each stage, eventually outputting the top 'N' of the proposed sites.
