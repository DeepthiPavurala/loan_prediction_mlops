from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


@dataclass
class Response:
    """Simple response container holding status code and parsed JSON body."""

    status_code: int
    body: Any


class RequestClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        logger.info("RequestClient initialized with host: %s", self.base_url)

    def _request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | list | None = None,
        params: dict[str, Any] | None = None,
    ) -> Response:
        url = f"{self.base_url}{path}"

        if params:
            clean_params = {key: value for key, value in params.items() if value is not None}

            query_string = urlencode(clean_params)

            if query_string:
                url = f"{url}?{query_string}"

        data = None
        headers = {"Content-Type": "application/json"}

        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")

        logger.info(">>> %s %s", method, url)
        if data:
            logger.debug(">>> body: %s", data[:500])

        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read()
                body = json.loads(raw) if raw else None
                logger.info("<<< %s %s", response.status, url)
                logger.debug("<<< body: %s", str(body)[:500])
                return Response(status_code=response.status, body=body)

        except urllib.error.HTTPError as exc:
            raw = exc.read()
            body = json.loads(raw) if raw else None
            logger.info("<<< %s %s", exc.code, url)
            logger.debug("<<< body: %s", str(body)[:500])
            return Response(status_code=exc.code, body=body)

    def get(self, path: str, params: dict[str, Any] | None = None) -> Response:
        return self._request(method="GET", path=path, params=params)

    def post(self, path: str, json: dict[str, Any] | list | None = None) -> Response:
        return self._request(method="POST", path=path, json_body=json)

    def patch(self, path: str, json: dict[str, Any] | None = None) -> Response:
        return self._request(method="PATCH", path=path, json_body=json)

    def delete(self, path: str) -> Response:
        return self._request(method="DELETE", path=path)
