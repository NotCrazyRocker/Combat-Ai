# Python WebSocket agent

Install the WebSocket client dependency:

```powershell
py -3 -m pip install websockets
```

Start the agent from the repository root:

```powershell
py -3 python\bridge\websocket_agent.py
```

The agent connects to the mod's WebSocket server at `ws://127.0.0.1:8765`, records complete observations and actions in `python/recordings/`, and sends a temporary scripted policy. The mod binds to localhost only. See `../protocol/observation.schema.json` for the observation payload contract.
