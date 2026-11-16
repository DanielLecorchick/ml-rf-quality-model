from io import BytesIO
import json
from pathlib import Path
import zipfile

import pytest

from rf_quality.ingestion.psws_client import ObservationQuery, download_observations, safe_unzip


def make_zip(name="S000028/OBS2022-12-09T00-00/ch0/test.h5", contents=b"binary"):
    b = BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr(name, contents)
    return b.getvalue()


class FakeResponse:
    def __init__(self, data):
        self.data = data
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False
    def raise_for_status(self):
        pass
    def iter_content(self, chunk_size):
        for i in range(0, len(self.data), 10):
            yield self.data[i:i + 10]


class FakeSession:
    def __init__(self, data):
        self.data = data
        self.params = None
        self.url = None
    def get(self, url, params, stream, timeout):
        self.url, self.params = url, params
        return FakeResponse(self.data)


def test_query_params_and_validation():
    q = ObservationQuery("2022-12-09", "2022-12-10", "S000028", frequency=10)
    assert q.params()["station_id"] == "S000028"
    assert q.params()["frequency"] == 10
    for broken in [
        ObservationQuery("not-a-date", "2022-12-10", "S000028"),
        ObservationQuery("2023-12-09", "2022-12-10", "S000028"),
        ObservationQuery("2022-12-09", "2022-12-10"),
        ObservationQuery("2022-12-09", "2022-12-10", "S000028", lat_min=20),
        ObservationQuery("2022-12-09", "2022-12-10", "BAD"),
        ObservationQuery("2022-12-09", "2022-12-10", "S000028", frequency=100),
    ]:
        with pytest.raises(ValueError):
            broken.params()
    q = ObservationQuery("2022-12-09", "2022-12-10", lat_min=32, lat_max=35,
                         lon_min=-88, lon_max=-84)
    assert q.params()["lat_min"] == 32


def test_download_streams_validates_and_manifests(tmp_path):
    body = make_zip()
    session = FakeSession(body)
    q = ObservationQuery("2022-12-09", "2022-12-10", "S000028")
    archive = download_observations(q, tmp_path, session=session)
    assert archive.read_bytes() == body
    assert session.params == q.params()
    manifest = json.loads(archive.with_suffix(".json").read_text())
    assert manifest["params"]["station_id"] == "S000028"
    assert manifest["bytes"] == len(body)
    dest = safe_unzip(archive, tmp_path / "out")
    assert (dest / "S000028/OBS2022-12-09T00-00/ch0/test.h5").read_bytes() == b"binary"
    # Cached ZIP does not issue a second request.
    assert download_observations(q, tmp_path, session=FakeSession(b"bad")) == archive


def test_invalid_download_is_rejected(tmp_path):
    q = ObservationQuery("2022-12-09", "2022-12-10", "S000028")
    with pytest.raises(ValueError, match="valid ZIP"):
        download_observations(q, tmp_path, session=FakeSession(b"<html>Not a ZIP</html>"))
    assert not list(tmp_path.glob("*.zip"))


def test_unsafe_archives_are_rejected(tmp_path):
    file = tmp_path / "malicious.zip"
    file.write_bytes(make_zip("../escape.txt"))
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        safe_unzip(file, tmp_path / "target")
    assert not (tmp_path / "escape.txt").exists()
    file.write_bytes(make_zip("large.txt", b"too big"))
    with pytest.raises(ValueError, match="allowed size"):
        safe_unzip(file, tmp_path / "target", max_total_bytes=3)
