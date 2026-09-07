# WebSocket performance regression fix

## Verification

```sh
cd /path/to/RoomBotSim/roombotsim
python -m py_compile tests/run_all.py
python tests/run_all.py
```

Results: `py_compile` exited 0; the full suite reported **23 passed, 0 failed**. The network performance checks reported:

```text
PASS performance: real network WebSocket connects and receives 6-person/3-robot state frames  frames=589 matching=589 errors=[]
PASS performance: 30 s wall advances >=29 s (6 people, 3 robots, network WebSocket client)  sim=30.11s wall=30.10s
```

## Network proof

`tests/run_all.py` reserves an ephemeral `127.0.0.1` TCP listener and passes that socket to an in-process-threaded `uvicorn.Server` serving the real `server.app`. It registers the performance runtime in `server.JOBS`, then a separate client thread uses `websockets.connect("ws://127.0.0.1:<ephemeral-port>/ws/<performance-job>")` and continuously `recv`s plus JSON-decodes frames for the measurement. The test asserts connection, received `state` frames containing exactly six people and three active robots, >=30 s wall time, and >=29 s simulation time. It does not use `TestClient`, direct route invocation, or `Runtime.state_since`; only the actual server `/ws/{job}` handler calls its runtime state publisher. Client, runtime, uvicorn thread, listener, and temporary job registration are cleaned up in `finally`.
