# Python endpoint

No third-party packages required for Phase 1. Start the TCP receiver from the repository root:

```powershell
py -3 python\bridge\receiver.py
```

The listener binds to `127.0.0.1:8765` and writes one observation object per line in `python/recordings/`. The mod uses the same host and port by default. See `../protocol/observation.schema.json` for the versioned payload contract.
