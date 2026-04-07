from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger("spam_app")


def _smtp_settings() -> dict:
    return {
        "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "username": os.getenv("SMTP_USERNAME", ""),
        "password": os.getenv("SMTP_PASSWORD", ""),
        "from_addr": os.getenv("SMTP_FROM", os.getenv("SMTP_USERNAME", "")),
        "to_addr": os.getenv("SMTP_TO", "spamdetectionsystem@gmail.com"),
        "use_tls": os.getenv("SMTP_TLS", "true").lower() != "false",
    }


def send_feedback_email(*, name: str, email: str, subject: str, message: str) -> bool:
    cfg = _smtp_settings()
    msg = EmailMessage()
    msg["Subject"] = f"Feedback: {subject}"
    msg["From"] = cfg["from_addr"] or email
    msg["To"] = cfg["to_addr"]
    body = (
        f"Time: {datetime.now().isoformat()}\n"
        f"From: {name} <{email}>\n"
        f"Subject: {subject}\n\n"
        f"{message}\n"
    )
    msg.set_content(body)

    if not cfg["username"] or not cfg["password"]:
        logger.warning("SMTP credentials not configured; writing feedback to file instead of sending email")
        _write_fallback(body)
        return False

    try:
        if cfg["use_tls"]:
            with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
                server.starttls(context=ssl.create_default_context())
                server.login(cfg["username"], cfg["password"])
                server.send_message(msg)
        else:
            with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=ssl.create_default_context()) as server:
                server.login(cfg["username"], cfg["password"])
                server.send_message(msg)
        logger.info("Feedback email delivered to %s", cfg["to_addr"])
        return True
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed sending feedback email: %s", exc)
        _write_fallback(body)
        return False


def _write_fallback(content: str) -> None:
    base = Path(__file__).resolve().parents[1]
    data_dir = base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    log_file = data_dir / "feedback_fallback.log"
    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n--- FEEDBACK ---\n")
        f.write(content)
        f.write("\n---------------\n")

