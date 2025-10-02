# auth_route.py

from typing import Callable
from fastapi import Request, Response, HTTPException
from fastapi.routing import APIRoute
from token_db import check, consume  # Import both helper functions

class OneTimeTokenRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_handler = super().get_route_handler()

        async def custom_handler(request: Request) -> Response:
            token = request.query_params.get("token")
            if not token:
                raise HTTPException(status_code=403, detail="Missing token")

            # Step 1: For any protected route, first check if the token is valid and unused.
            # The `check` function does NOT consume the token.
            is_valid = await check(token)
            if not is_valid:
                raise HTTPException(status_code=403, detail="Invalid or expired token")

            # Step 2: If the request is for the main registration endpoint,
            # we consume the token to make it a one-time action.
            if request.url.path.startswith("/register"):
                was_consumed = await consume(token)
                if not was_consumed:
                    # This could happen in a rare race condition.
                    raise HTTPException(status_code=403, detail="Token has just been used")

            # If the checks passed, proceed to the actual endpoint (e.g., /validate_token or /register)
            return await original_handler(request)

        return custom_handler