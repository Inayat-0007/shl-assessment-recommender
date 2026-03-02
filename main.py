"""
Entry point for the SHL SmartMatch recommendation API.

I'm using uvicorn with reload=True during development so I don't have to
manually restart the server every time I tweak something in the engine.
For production (Render deployment), the Procfile handles the start command.

    $ python main.py

- Mohammad Inayat Hussain
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=True,
    )
