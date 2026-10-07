"""Components for alerting on recent change from satellite image time series.

This targets the case where a model should flag changes within weeks of when they
become observable, by inputting a time series whose most recent images are frequent
(e.g. weekly mosaics), optionally preceded by infrequent images (e.g. quarterly
mosaics) that provide historical context. It is not intended for comparing conditions
between two distant time periods (e.g. one year versus another), where a model that
simply inputs a regular time series spanning both periods is sufficient.

During training, each example is labeled with the date at which a change first becomes
observable. Several time series ending at different offsets after that date are
materialized as separate dataset layers (see slots.py), and ChangeTimeSeriesSampler
(see sampler.py) picks one of them per example and derives per-pixel targets for the
change category and for the timestep at which the change appears.
"""
