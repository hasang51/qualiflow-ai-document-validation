from __future__ import annotations

from pathlib import Path

from app.services.storage.local import LocalStorageBackend


def test_local_storage_put_get_exists_delete(tmp_path: Path):
    backend = LocalStorageBackend(root=tmp_path)
    key = "uploads/demo.pdf"
    data = b"%PDF-1.4 test"
    backend.put_bytes(key, data, content_type="application/pdf")
    assert backend.exists(key)
    assert backend.get_bytes(key) == data
    backend.delete(key)
    assert not backend.exists(key)


def test_s3_storage_mocked(monkeypatch):
    calls: dict[str, object] = {}

    class FakeBody:
        def read(self):
            return b"payload"

    class FakeClient:
        def put_object(self, **kwargs):
            calls["put"] = kwargs

        def get_object(self, **kwargs):
            calls["get"] = kwargs
            return {"Body": FakeBody()}

        def head_object(self, **kwargs):
            calls["head"] = kwargs

        def delete_object(self, **kwargs):
            calls["delete"] = kwargs

    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: FakeClient())

    from config.settings import Settings, reset_settings_cache
    from app.services.storage.factory import reset_storage_backend_cache

    reset_settings_cache()
    monkeypatch.setenv("OBJECT_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("S3_BUCKET_NAME", "qualiflow")
    reset_settings_cache()

    from app.services.storage.s3 import S3StorageBackend

    reset_storage_backend_cache()
    backend = S3StorageBackend()
    backend.put_bytes("k1", b"abc")
    assert backend.get_bytes("k1") == b"payload"
    assert backend.exists("k1") is True
    backend.delete("k1")
    assert "put" in calls

    reset_settings_cache()
    reset_storage_backend_cache()
