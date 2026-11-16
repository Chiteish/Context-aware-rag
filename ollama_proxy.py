
import os

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

PROXY_TOKEN = os.environ.get("OLLAMA_PROXY_TOKEN")

if not PROXY_TOKEN:
    raise RuntimeError("OLLAMA_PROXY_TOKEN is not configured")


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE"]
)
async def proxy(
    path: str,
    request: Request,
    authorization: str | None = Header(default=None)
):
    if authorization != f"Bearer {PROXY_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    url = f"http://127.0.0.1:11434/{path}"

    body = await request.body()

    async with httpx.AsyncClient(timeout=None) as client:
        upstream = await client.send(
            client.build_request(
                request.method,
                url,
                params=request.query_params,
                content=body,
                headers={
                    "Content-Type": request.headers.get(
                        "content-type", "application/json"
                    )
                },
            ),
            stream=True,
        )

        async def response_stream():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            response_stream(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type")
        )
