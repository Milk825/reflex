import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path

from analysis.autonomy.core.bridge import BridgeServer


async def request_once(socket_path: str) -> float:
    payload = {
        "action_id": "bench",
        "hook_type": "before_action",
        "context": {"toolName": "read", "params": {"path": "x"}},
        "timestamp": time.time(),
    }
    body = json.dumps(payload).encode("utf-8")

    started = time.perf_counter()
    reader, writer = await asyncio.open_unix_connection(socket_path)
    writer.write(
        (
            "POST /before_action HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "\r\n"
        ).encode("utf-8")
        + body
    )
    await writer.drain()

    await reader.readline()  # status
    headers = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        k, v = line.decode("utf-8").split(":", 1)
        headers[k.strip().lower()] = v.strip()

    content_length = int(headers.get("content-length", "0"))
    if content_length:
        await reader.readexactly(content_length)

    writer.close()
    await writer.wait_closed()
    elapsed_ms = (time.perf_counter() - started) * 1000
    return elapsed_ms


async def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        socket_path = str(Path(td) / "bridge.sock")
        server = BridgeServer(socket_path, handler_timeout_ms=90)
        await server.start()

        try:
            # warmup
            for _ in range(10):
                await request_once(socket_path)

            samples = [await request_once(socket_path) for _ in range(100)]
            mean_ms = statistics.mean(samples)
            p95_ms = statistics.quantiles(samples, n=20)[18]
            print(json.dumps({"samples": len(samples), "mean_ms": mean_ms, "p95_ms": p95_ms}, indent=2))
        finally:
            await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
