from enum import Enum, auto
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse


class DataSourceType(Enum):
    LOCAL = auto()
    HTTP = auto()


class DataSource:
    def __init__(self, name: str, ds_type: DataSourceType, uri: str):
        self.name = name
        self.ds_type = ds_type
        self.uri = uri

    def __str__(self):
        return f"{self.name} : {self.ds_type} : {self.uri}"

    @property
    def uri(self):
        return self._uri

    @uri.setter
    def uri(self, value: str):
        if self.ds_type == DataSourceType.LOCAL:
            path = Path(value)
            if not path.is_absolute():
                raise ValueError(
                    "uri field for a LOCAL DataSource must be a valid absolute path"
                )
        elif self.ds_type is DataSourceType.HTTP:
            url = urlparse(value)
            if not url.netloc:
                raise ValueError(
                    "uri field for an HTTP DataSource must be a valid URL"
                )
        self._uri = value


_PHYSIONET_DATA_SOURCE = DataSource(
    "physionet",
    DataSourceType.HTTP,
    "https://physionet.org/files/",
)

# Dict of configured data sources
_data_sources: Dict[str, DataSource] = {"physionet": _PHYSIONET_DATA_SOURCE}


def show_data_sources():
    """
    Displays all configured data sources
    """
    print("Data sources:")
    for _, ds in _data_sources.items():
        print(ds)


def add_data_source(ds: DataSource):
    """
    Add a data source to the set of configured data sources
    """
    if ds.name in _data_sources:
        raise ValueError(
            f"There is already a configured data source with name: {ds.name}"
        )

    _data_sources[ds.name] = ds


def remove_data_source(ds_name: str):
    """
    Remove a data source from the set of configured data sources
    """
    del _data_sources[ds_name]


def reset_data_sources(keep_pn: bool = False):
    """
    Reset all configured data sources

    Parameters
    ----------

    keep_pn : bool
        If True, keep the default physionet data source.

    """
    _data_sources.clear()
    if keep_pn:
        _data_sources["physionet"] = _PHYSIONET_DATA_SOURCE
