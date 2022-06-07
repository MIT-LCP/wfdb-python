import platform
import pytest

from wfdb.io import (
    DataSource,
    DataSourceType,
    add_data_source,
    remove_data_source,
    reset_data_sources,
)

from wfdb.io.datasource import _data_sources

LOCAL_PATH = (
    "C:\\Users\\Public\\data"
    if platform.system() == "Windows"
    else "/bigdata/smalldata"
)


class TestDataSource:
    def test_create_valid_local_ds(self):
        ds = DataSource(
            name="localds",
            ds_type=DataSourceType.LOCAL,
            uri=LOCAL_PATH,
        )
        assert ds

    def test_create_invalid_local_ds(self):
        with pytest.raises(ValueError):
            DataSource(
                name="localds",
                ds_type=DataSourceType.LOCAL,
                uri="notabsolute",
            )

    def test_create_valid_http_ds(self):
        ds = DataSource(
            name="httpds",
            ds_type=DataSourceType.HTTP,
            uri="http://bigdata.com",
        )
        assert ds.uri == "http://bigdata.com"

    def test_create_invalid_http_ds(self):
        with pytest.raises(ValueError):
            DataSource(
                name="httpds",
                ds_type=DataSourceType.HTTP,
                uri="www.bigdata.com",
            )

    def test_add_reset_ds(self):
        ds = DataSource(
            name="localds",
            ds_type=DataSourceType.LOCAL,
            uri=LOCAL_PATH,
        )
        add_data_source(ds)
        assert len(_data_sources) == 2
        assert _data_sources[ds.name] == ds
        # We rely on reset_data_sources for test cleanup.
        reset_data_sources(keep_pn=True)
        assert len(_data_sources) == 1

    def test_add_multiple_ds(self):
        ds1 = DataSource(
            name="localds",
            ds_type=DataSourceType.LOCAL,
            uri=LOCAL_PATH,
        )
        add_data_source(ds1)
        ds2 = DataSource(
            name="anotherlocalds",
            ds_type=DataSourceType.LOCAL,
            uri=LOCAL_PATH,
        )
        add_data_source(ds2)

        assert len(_data_sources) == 3
        assert _data_sources[ds1.name] == ds1
        assert _data_sources[ds2.name] == ds2
        reset_data_sources(keep_pn=True)

    def test_remove_ds(self):
        ds = DataSource(
            name="localds",
            ds_type=DataSourceType.LOCAL,
            uri=LOCAL_PATH,
        )
        add_data_source(ds)
        remove_data_source("localds")
        assert len(_data_sources) == 1

    def test_unique_ds_names(self):
        ds = DataSource(
            name="localds",
            ds_type=DataSourceType.LOCAL,
            uri=LOCAL_PATH,
        )
        add_data_source(ds)
        # Cannot set multiple data sources with the same name
        with pytest.raises(ValueError):
            add_data_source(ds)
        reset_data_sources(keep_pn=True)
