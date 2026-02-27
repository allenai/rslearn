## rslearn.data_sources.eurocrops.EuroCrops

This data source is for EuroCrops vector data (v11).

See https://zenodo.org/records/14094196 for details.

While the source data is split into country-level files, this data source uses one item
per year for simplicity. So each item corresponds to all of the country-level files for
that year.

Note that the RO_ny.zip file is not used.

The vector features should have `EC_hcat_c` and `EC_hcat_n` properties indicating the
HCAT category code and name respectively.

### Configuration

There is no data-source-specific configuration.
