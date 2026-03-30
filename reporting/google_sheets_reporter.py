import importlib
import os
from datetime import datetime, timezone
from typing import Any

from google.oauth2.service_account import Credentials


class GoogleSheetsEpochReporter:
    _SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    _HEADER = [
        "timestamp_utc",
        "experiment_name",
        "epoch",
        "accuracy",
        "roc_auc",
        "average_precision",
        "total_steps",
    ]

    def __init__(
        self,
        spreadsheet_id: str,
        experiment_name: str,
        credentials_path: str | None = None,
    ):
        self._spreadsheet_id = spreadsheet_id
        self._experiment_name = experiment_name
        self._credentials_path = credentials_path
        self._gspread = self._import_gspread()
        self._worksheet = self._open_or_create_worksheet()
        self._ensure_header()

    def append_epoch_result(
        self,
        epoch: int,
        accuracy: float,
        roc_auc: float,
        average_precision: float,
        total_steps: int,
    ) -> None:
        row = [
            datetime.now(timezone.utc).isoformat(),
            self._experiment_name,
            epoch,
            accuracy,
            roc_auc,
            average_precision,
            total_steps,
        ]
        self._worksheet.append_row(row, value_input_option="RAW")

    def _open_or_create_worksheet(self) -> Any:
        spreadsheet = self._open_spreadsheet()
        worksheet_name = self._build_worksheet_name(self._experiment_name)
        try:
            return spreadsheet.worksheet(worksheet_name)
        except self._gspread.WorksheetNotFound:
            return spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=20)

    def _open_spreadsheet(self) -> Any:
        client = self._build_client()
        return client.open_by_key(self._spreadsheet_id)

    def _build_client(self) -> Any:
        credentials_path = self._resolve_credentials_path()
        credentials = Credentials.from_service_account_file(
            credentials_path,
            scopes=self._SCOPES,
        )
        return self._gspread.authorize(credentials)

    def _resolve_credentials_path(self) -> str:
        if self._credentials_path:
            return self._credentials_path

        env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if env_path:
            return env_path

        raise ValueError(
            "Google Sheets reporting requires credentials. Set --google_sheets_credentials_path or GOOGLE_APPLICATION_CREDENTIALS."
        )

    def _ensure_header(self) -> None:
        first_row = self._worksheet.row_values(1)
        if not first_row:
            self._worksheet.append_row(self._HEADER, value_input_option="RAW")

    @staticmethod
    def _build_worksheet_name(experiment_name: str) -> str:
        invalid_chars = set("[]:*?/\\")
        sanitized = "".join("_" if char in invalid_chars else char for char in experiment_name).strip()
        return (sanitized or "experiment")[:100]

    @staticmethod
    def _import_gspread() -> Any:
        try:
            gspread = importlib.import_module('gspread')
        except ImportError as exc:
            raise ImportError(
                "gspread is required for Google Sheets reporting. Install it with 'pip install gspread'."
            ) from exc
        return gspread
