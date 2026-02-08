"""
Snowflake Agentic RAG tool: query Snowflake Cortex (RAG/COMPLETE) from the voice agent.
Also writes conversation (user + assistant) to Snowflake when SNOWFLAKE_CHAT_TABLE is set.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger("snowflake-rag-tool")


def _get_connection_params() -> Optional[dict[str, Any]]:
    """Build Snowflake connection params from env. Returns None if not configured."""
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    user = os.getenv("SNOWFLAKE_USER")
    if not account or not user:
        return None
    password = os.getenv("SNOWFLAKE_PASSWORD")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "")
    database = os.getenv("SNOWFLAKE_DATABASE", "")
    schema = os.getenv("SNOWFLAKE_SCHEMA", "")
    role = os.getenv("SNOWFLAKE_ROLE")
    connect_params: dict[str, Any] = {
        "account": account.strip(),
        "user": user.strip(),
        "warehouse": warehouse.strip() or None,
        "database": database.strip() or None,
        "schema": schema.strip() or None,
    }
    if password:
        connect_params["password"] = password.strip().strip('"').strip("'")
    elif os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH"):
        with open(os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"], "rb") as f:
            pkey = serialization.load_pem_private_key(
                f.read(),
                password=os.getenv("SNOWFLAKE_PRIVATE_KEY_PASS") or None,
                backend=default_backend(),
            )
        connect_params["private_key"] = pkey
    else:
        return None
    if role:
        connect_params["role"] = role.strip()
    return connect_params


def _write_chat_to_snowflake_sync(
    session_id: str,
    participant_id: str,
    role: str,
    message: str,
    agent_name: str,
) -> None:
    """Write one chat row to Snowflake. No-op if SNOWFLAKE_CHAT_TABLE is not set."""
    table = (os.getenv("SNOWFLAKE_CHAT_TABLE") or "").strip()
    if not table or not message or not message.strip():
        return
    if not re.match(r"^[a-zA-Z0-9_]+$", table):
        logger.warning("SNOWFLAKE_CHAT_TABLE must be a single identifier (letters, numbers, underscore)")
        return
    params = _get_connection_params()
    if not params:
        logger.warning("Snowflake chat logging skipped: connection not configured")
        return
    conn = None
    try:
        db = os.getenv("SNOWFLAKE_CHAT_DATABASE") or params.get("database")
        sch = os.getenv("SNOWFLAKE_CHAT_SCHEMA") or params.get("schema")
        if db:
            params = {**params, "database": db}
        if sch:
            params = {**params, "schema": sch}
        conn = snowflake.connector.connect(**params)
        cursor = conn.cursor()
        created_at = datetime.now(timezone.utc).isoformat()
        # Table expected: session_id, participant_id, role, message, agent_name, created_at
        sql = (
            f'INSERT INTO "{table}" (session_id, participant_id, role, message, agent_name, created_at) '
            "VALUES (%s, %s, %s, %s, %s, %s)"
        )
        cursor.execute(
            sql,
            (session_id, participant_id, role, (message or "").strip()[:65535], agent_name or "", created_at),
        )
        conn.commit()
        cursor.close()
    except Exception as e:
        logger.exception("Snowflake chat write error: %s", e)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


async def write_chat_to_snowflake(
    session_id: str,
    participant_id: str,
    role: str,
    message: str,
    agent_name: str = "",
) -> None:
    """Write a conversation turn to Snowflake (async). No-op if SNOWFLAKE_CHAT_TABLE unset."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: _write_chat_to_snowflake_sync(session_id, participant_id, role, message, agent_name),
    )


def _run_snowflake_sync(question: str, model: str, system_instruction: Optional[str], custom_sql: Optional[str]) -> str:
    """Run Snowflake Cortex COMPLETE (or custom RAG SQL) in a sync way. Call from async via to_thread."""
    conn = None
    try:
        account = os.getenv("SNOWFLAKE_ACCOUNT")
        user = os.getenv("SNOWFLAKE_USER")
        password = os.getenv("SNOWFLAKE_PASSWORD")
        warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "")
        database = os.getenv("SNOWFLAKE_DATABASE", "")
        schema = os.getenv("SNOWFLAKE_SCHEMA", "")
        role = os.getenv("SNOWFLAKE_ROLE")

        if not account or not user:
            return "Snowflake is not configured (set SNOWFLAKE_ACCOUNT and SNOWFLAKE_USER)."

        connect_params: dict[str, Any] = {
            "account": account.strip(),
            "user": user.strip(),
            "warehouse": warehouse.strip() or None,
            "database": database.strip() or None,
            "schema": schema.strip() or None,
        }
        if password:
            connect_params["password"] = password.strip().strip('"').strip("'")
        elif os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH"):
            with open(os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"], "rb") as f:
                pkey = serialization.load_pem_private_key(
                    f.read(),
                    password=os.getenv("SNOWFLAKE_PRIVATE_KEY_PASS") or None,
                    backend=default_backend(),
                )
            connect_params["private_key"] = pkey
        else:
            return "Snowflake credentials missing (set SNOWFLAKE_PASSWORD or SNOWFLAKE_PRIVATE_KEY_PATH)."

        if role:
            connect_params["role"] = role.strip()

        conn = snowflake.connector.connect(**connect_params)
        cursor = conn.cursor()

        if custom_sql:
            # User-provided SQL (e.g. CALL my_rag_proc(?) or SELECT ... CORTEX.COMPLETE(...))
            # Assume single ? placeholder for the question
            cursor.execute(custom_sql, (question,))
        else:
            # Default: SNOWFLAKE.CORTEX.COMPLETE(model, conversation_array, options)
            # With array + options, response is JSON: {"choices":[{"messages":"..."}], ...}
            prompt_arr = [{"role": "user", "content": question}]
            if system_instruction:
                prompt_arr.insert(0, {"role": "system", "content": system_instruction})
            prompt_json = json.dumps(prompt_arr)
            sql = "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, PARSE_JSON(%s), {}) AS response"
            cursor.execute(sql, (model, prompt_json))

        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            return "No response from Snowflake."
        out = row[0]
        if out is None:
            return "Empty response from Snowflake."
        out_str = str(out).strip()
        # When using array+options, COMPLETE returns JSON; extract the message text
        if out_str.startswith("{"):
            try:
                data = json.loads(out_str)
                choices = data.get("choices") or []
                if choices and isinstance(choices[0], dict) and "messages" in choices[0]:
                    return (choices[0]["messages"] or "").strip()
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
        return out_str
    except Exception as e:
        logger.exception("Snowflake RAG error: %s", e)
        return f"I couldn't get an answer from the knowledge base: {e!s}."
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


async def get_snowflake_rag_response(
    question: str,
    *,
    model: Optional[str] = None,
    system_instruction: Optional[str] = None,
    custom_sql: Optional[str] = None,
) -> str:
    """
    Run agentic RAG on Snowflake: ask a natural-language question and get an answer
    using Snowflake Cortex (COMPLETE) or a custom RAG stored procedure/SQL.

    Set in .env:
      SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD (or private key),
      optionally SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA, SNOWFLAKE_ROLE.
    Optional: SNOWFLAKE_RAG_SQL for custom SQL (one ? for the question).
    Optional: SNOWFLAKE_RAG_SYSTEM_INSTRUCTION for default system prompt.
    """
    model = model or os.getenv("SNOWFLAKE_RAG_MODEL", "mistral-large")
    system_instruction = system_instruction or os.getenv("SNOWFLAKE_RAG_SYSTEM_INSTRUCTION")
    custom_sql = custom_sql or os.getenv("SNOWFLAKE_RAG_SQL")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: _run_snowflake_sync(question, model, system_instruction, custom_sql),
    )
