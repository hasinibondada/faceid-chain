"""Vercel serverless entrypoint.

Serves the FastAPI ASGI app from `app.web.server`. Vercel's Python runtime runs
this file directly (ASGI object named `app`); `vercel.json` routes every path to
`/api/index`. No uvicorn is required on Vercel.
"""
import app.web.server as server

app = server.app