# LAN handoff QR

`qr.png` currently encodes `http://10.104.4.240:8000/` (the detected LAN address while TASK-07 was completed).

If the laptop changes networks or uses another port, regenerate it from the `walkin/` directory:

```sh
.venv/bin/python static/generate_qr.py http://YOUR.LAN.IP:8000/
```

The generator needs only the standard library and Pillow, which the project already requires, and it refuses loopback targets. Start the server on the LAN interface (for example `uvicorn server:app --host 0.0.0.0 --port 8000`) and have the phone join the same network.
