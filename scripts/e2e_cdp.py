"""A small Chrome DevTools Protocol client.

Enough of CDP to drive a real browser: launch it, click at real coordinates,
type real keystrokes, read the DOM back and collect console and network errors.
Selenium and Playwright both do this better, but each pulls in a driver stack
and a managed browser download. This is one file with one dependency
(``websockets``), which keeps an end-to-end check from becoming a build system.
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import websockets

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "chromium",
    "chromium-browser",
)


def find_chrome() -> str:
    """Return a usable Chrome/Chromium binary, or raise with a clear message."""
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise RuntimeError(
        "no Chrome or Chromium found. Install one, or set CHROME_PATH to the binary."
    )


class ChromeProcess:
    """Headless Chrome with the DevTools endpoint open, cleaned up on exit."""

    def __init__(self, port: int = 9222, window: str = "1280,1400") -> None:
        self.port = port
        self.window = window
        self._proc: subprocess.Popen | None = None
        self._profile: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> ChromeProcess:
        self._profile = tempfile.TemporaryDirectory(prefix="medicare-e2e-")
        self._proc = subprocess.Popen(
            [
                find_chrome(),
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                f"--remote-debugging-port={self.port}",
                f"--user-data-dir={self._profile.name}",
                f"--window-size={self.window}",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._profile is not None:
            self._profile.cleanup()


class Browser:
    """One CDP session against one page."""

    def __init__(self, ws_url: str) -> None:
        self.ws_url = ws_url
        self.console: list[tuple[str, str]] = []
        self.errors: list[str] = []
        self.failed: list[str] = []
        self._id = 0
        self._pending: dict[int, asyncio.Future] = {}

    @classmethod
    async def attach(cls, port: int = 9222, tries: int = 60) -> Browser:
        """Wait for Chrome's DevTools endpoint and attach to its first page."""
        last: Exception | None = None
        for _ in range(tries):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/json") as response:
                    tabs = json.loads(response.read())
                page = next(tab for tab in tabs if tab["type"] == "page")
                return cls(page["webSocketDebuggerUrl"])
            except (urllib.error.URLError, StopIteration, ConnectionError, OSError) as exc:
                last = exc
                await asyncio.sleep(0.5)
        raise RuntimeError(f"could not attach to Chrome on port {port}: {last}")

    async def __aenter__(self) -> Browser:
        self.ws = await websockets.connect(self.ws_url, max_size=80 * 1024 * 1024)
        self._pump = asyncio.create_task(self._reader())
        for domain in ("Page", "Runtime", "Log", "Network"):
            await self.send(f"{domain}.enable")
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self._pump.cancel()
        await self.ws.close()

    async def _reader(self) -> None:
        async for raw in self.ws:
            message = json.loads(raw)
            if "id" in message and message["id"] in self._pending:
                self._pending.pop(message["id"]).set_result(message)
                continue
            self._record(message.get("method"), message.get("params", {}))

    def _record(self, method: str | None, params: dict) -> None:
        if method == "Runtime.consoleAPICalled":
            text = " ".join(
                str(arg.get("value", arg.get("description", ""))) for arg in params.get("args", [])
            )
            self.console.append((params.get("type", ""), text))
            if params.get("type") == "error":
                self.errors.append(text)
        elif method == "Runtime.exceptionThrown":
            self.errors.append(str(params.get("exceptionDetails", {}).get("text", "")))
        elif method == "Log.entryAdded" and params.get("entry", {}).get("level") == "error":
            self.errors.append(params["entry"].get("text", ""))
        elif method == "Network.loadingFailed":
            self.failed.append(params.get("errorText", ""))
        elif method == "Network.responseReceived":
            response = params.get("response", {})
            if response.get("status", 0) >= 400:
                self.failed.append(f"HTTP {response['status']} {response.get('url', '')[:90]}")

    async def send(self, method: str, **params: object) -> dict:
        self._id += 1
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[self._id] = future
        await self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        message = await asyncio.wait_for(future, timeout=180)
        if "error" in message:
            raise RuntimeError(f"{method}: {message['error']}")
        return message.get("result", {})

    async def js(self, expression: str) -> object:
        """Evaluate an expression in the page and return its value."""
        result = await self.send(
            "Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True
        )
        if result.get("exceptionDetails"):
            raise RuntimeError(f"page threw: {result['exceptionDetails'].get('text')}")
        return result["result"].get("value")

    async def goto(self, url: str) -> None:
        await self.send("Page.navigate", url=url)
        await asyncio.sleep(1.0)

    async def wait_for(self, expression: str, timeout: float = 180, label: str = "") -> None:
        """Poll an expression until it is truthy."""
        for _ in range(int(timeout * 2)):
            try:
                if await self.js(expression):
                    return
            except RuntimeError:
                pass
            await asyncio.sleep(0.5)
        raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {label or expression}")

    async def centre(self, selector: str) -> tuple[float, float]:
        box = await self.js(
            f"""(() => {{
                const el = document.querySelector({selector!r});
                if (!el) return null;
                el.scrollIntoView({{block: 'center'}});
                const r = el.getBoundingClientRect();
                return {{x: r.left + r.width / 2, y: r.top + r.height / 2}};
            }})()"""
        )
        if not box:
            raise RuntimeError(f"element not found: {selector}")
        return box["x"], box["y"]

    async def click(self, selector: str) -> None:
        """Click the centre of an element with real mouse events."""
        x, y = await self.centre(selector)
        await self.click_at(x, y)

    async def click_at(self, x: float, y: float) -> None:
        for kind in ("mousePressed", "mouseReleased"):
            await self.send(
                "Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1
            )
        await asyncio.sleep(0.25)

    async def type_text(self, selector: str, text: str) -> None:
        await self.click(selector)
        await self.js(f"document.querySelector({selector!r}).value = ''")
        await self.send("Input.insertText", text=text)
        await asyncio.sleep(0.15)

    async def press_enter(self) -> None:
        """Press Enter.

        The ``char`` event is the point: without it a browser does not treat the
        key as implicit form submission, and the form never submits.
        """
        common = {
            "key": "Enter",
            "code": "Enter",
            "windowsVirtualKeyCode": 13,
            "nativeVirtualKeyCode": 13,
        }
        await self.send("Input.dispatchKeyEvent", type="rawKeyDown", **common)
        await self.send("Input.dispatchKeyEvent", type="char", text="\r", **common)
        await self.send("Input.dispatchKeyEvent", type="keyUp", **common)
        await asyncio.sleep(0.2)

    async def emulate_phone(self, width: int = 390, height: int = 844) -> None:
        await self.send(
            "Emulation.setDeviceMetricsOverride",
            width=width,
            height=height,
            deviceScaleFactor=2,
            mobile=True,
        )

    async def clear_emulation(self) -> None:
        await self.send("Emulation.clearDeviceMetricsOverride")

    async def screenshot(self, path: Path) -> None:
        data = await self.send("Page.captureScreenshot", format="png")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(data["data"]))
