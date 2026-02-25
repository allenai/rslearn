## Data Sources

This document lists the built-in data sources in rslearn. Each data source links to a
detailed page with configuration options and available bands. See
[Dataset Configuration](DatasetConfig.md) for how to use these in a dataset config file.

### General-Purpose

| Data Source | Description |
|---|---|
| [local_files.LocalFiles](data_sources/local_files_LocalFiles.md) | Ingest from local raster or vector files |

### Remote Sensing Images

#### Sentinel-2

| Data Source | Provider | Notes |
|---|---|---|
| [planetary_computer.Sentinel2](data_sources/planetary_computer_Sentinel2.md) | Microsoft Planetary Computer | L2A COGs, direct materialization |
| [aws_open_data.Sentinel2](data_sources/aws_open_data_Sentinel2.md) | AWS (Element 84) | L1C and L2A |
| [aws_sentinel2_element84.Sentinel2](data_sources/aws_sentinel2_element84_Sentinel2.md) | AWS (Element 84) | L2A COGs, direct materialization |
| [copernicus.Sentinel2](data_sources/copernicus.md#rslearndatasourcescopernicussentinel2) | ESA Copernicus OData API | L1C and L2A |
| [earthdaily.Sentinel2](data_sources/earthdaily_Sentinel2.md) | EarthDaily | L2A, requires EarthDaily credentials |
| [gcp_public_data.Sentinel2](data_sources/gcp_public_data_Sentinel2.md) | Google Cloud Storage | L1C scenes |

#### Sentinel-1

| Data Source | Provider | Notes |
|---|---|---|
| [planetary_computer.Sentinel1](data_sources/planetary_computer_Sentinel1.md) | Microsoft Planetary Computer | terrain-corrected COGs, direct materialization |
| [aws_sentinel1.Sentinel1](data_sources/aws_sentinel1_Sentinel1.md) | AWS (Sinergise) | GRD IW DV |
| [copernicus.Sentinel1](data_sources/copernicus.md#rslearndatasourcescopernicussentinel1) | ESA Copernicus OData API | IW GRDH |

#### Landsat

| Data Source | Provider | Notes |
|---|---|---|
| [aws_landsat.LandsatOliTirs](data_sources/aws_landsat_LandsatOliTirs.md) | AWS (USGS) | Level-1, all bands |
| [planetary_computer.LandsatC2L2](data_sources/planetary_computer_LandsatC2L2.md) | Microsoft Planetary Computer | Collection 2 Level-2, direct materialization |
| [usgs_landsat.LandsatOliTirs](data_sources/usgs_landsat_LandsatOliTirs.md) | USGS M2M API | Requires M2M access |

#### HLS (Harmonized Landsat Sentinel-2)

| Data Source | Provider | Notes |
|---|---|---|
| [planetary_computer.Hls2S30](data_sources/planetary_computer_Hls2.md#rslearndatasourcesplanetary_computerhls2s30) | Microsoft Planetary Computer | HLS v2 Sentinel-2 (30m) |
| [planetary_computer.Hls2L30](data_sources/planetary_computer_Hls2.md#rslearndatasourcesplanetary_computerhls2l30) | Microsoft Planetary Computer | HLS v2 Landsat (30m) |

#### NAIP

| Data Source | Provider | Notes |
|---|---|---|
| [aws_open_data.Naip](data_sources/aws_open_data_Naip.md) | AWS (Esri) | Requester pays, requires AWS credentials |
| [planetary_computer.Naip](data_sources/planetary_computer_Naip.md) | Microsoft Planetary Computer | Direct materialization |

#### Other

| Data Source | Provider | Notes |
|---|---|---|
| [copernicus.Copernicus](data_sources/copernicus.md) | ESA Copernicus OData API | Generic product access |
| [google_earth_engine.GEE](data_sources/google_earth_engine_GEE.md) | Google Earth Engine | Generic ee.ImageCollection access |
| [aws_google_satellite_embedding_v1.GoogleSatelliteEmbeddingV1](data_sources/aws_google_satellite_embedding_v1.md) | AWS Open Data | Google Satellite Embedding v1 |
| [google_earth_engine.GoogleSatelliteEmbeddings](data_sources/google_earth_engine_GoogleSatelliteEmbeddings.md) | Google Earth Engine | Google Satellite Embedding v1, requires GEE credentials |
| [planetary_computer.PlanetaryComputer](data_sources/planetary_computer_PlanetaryComputer.md) | Microsoft Planetary Computer | Generic collection access |
| [planetary_computer.Sentinel3SlstrLST](data_sources/planetary_computer_Sentinel3SlstrLST.md) | Microsoft Planetary Computer | Land Surface Temperature, requires ingestion |
| [xyz_tiles.XyzTiles](data_sources/xyz_tiles_XyzTiles.md) | Any XYZ tile server | Web slippy tiles |

### Experimental

These data sources are still experimental and may have incomplete functionality.

- `rslearn.data_sources.planet.Planet`
- `rslearn.data_sources.planet_basemap.PlanetBasemap`

### Other Raster Data

#### Elevation

| Data Source | Description |
|---|---|
| [hf_srtm.SRTM](data_sources/hf_srtm_SRTM.md) | SRTM elevation (~30-90m), served from Hugging Face |
| [planetary_computer.CopDemGlo30](data_sources/planetary_computer_CopDemGlo30.md) | Copernicus DEM GLO-30 (30m) |

#### Climate / Weather

| Data Source | Description |
|---|---|
| [climate_data_store.ERA5Land](data_sources/climate_data_store.md) | ERA5 Land from Copernicus CDS (base class, monthly means, hourly) |
| [earthdatahub.ERA5LandDailyUTCv1](data_sources/earthdatahub_ERA5LandDailyUTCv1.md) | ERA5-Land daily UTC from EarthDataHub |

#### Land Cover / Crop

| Data Source | Description |
|---|---|
| [worldcover.WorldCover](data_sources/worldcover_WorldCover.md) | ESA WorldCover 2021 (10m) |
| [worldcereal.WorldCereal](data_sources/worldcereal_WorldCereal.md) | ESA WorldCereal 2021 crop map |
| [usda_cdl.CDL](data_sources/usda_cdl_CDL.md) | USDA Cropland Data Layer (US only) |

#### Soil

| Data Source | Description |
|---|---|
| [soilgrids.SoilGrids](data_sources/soilgrids_SoilGrids.md) | ISRIC SoilGrids via WCS (~250m) |
| [soildb.SoilDB](data_sources/soildb_SoilDB.md) | OpenLandMap SoilDB (30m) |

#### Population

| Data Source | Description |
|---|---|
| [worldpop.WorldPop](data_sources/worldpop_WorldPop.md) | WorldPop Constrained 2020 (100m) |

### Vector Data

| Data Source | Description |
|---|---|
| [openstreetmap.OpenStreetMap](data_sources/openstreetmap_OpenStreetMap.md) | OpenStreetMap features from PBF files |
| [eurocrops.EuroCrops](data_sources/eurocrops_EuroCrops.md) | EuroCrops v11 agricultural parcels |
