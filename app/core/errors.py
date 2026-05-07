from app.schemas.errors import ErrorCode, ErrorResponse, SafeErrorDetails


class AppError(Exception):
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        *,
        status_code: int,
        details: SafeErrorDetails | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def to_response(self, *, request_id: str | None = None) -> ErrorResponse:
        return ErrorResponse(
            error_code=self.error_code,
            message=self.message,
            request_id=request_id,
            details=self.details,
        )
