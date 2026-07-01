# Transport Stub Regeneration

`services_pb2.py` and `services_pb2_grpc.py` are generated from
[`services.proto`](/F:/DataImpact/lerobot/src/lerobot/transport/services.proto).

When `services.proto` changes, regenerate both checked-in files from the repo
root:

```powershell
uv sync --locked --extra dev
uv run python -m grpc_tools.protoc -I src --python_out=src --grpc_python_out=src src/lerobot/transport/services.proto
```

Notes:

- `grpcio-tools` is already declared in the repo's `dev` extra in
  [pyproject.toml](/F:/DataImpact/lerobot/pyproject.toml).
- Do not hand-edit `services_pb2.py` or `services_pb2_grpc.py`; rerun the
  command above instead.
