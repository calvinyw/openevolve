"""
Codex CLI interface for LLM calls.

This backend routes OpenEvolve prompts through the local `codex exec` command instead of
calling a remote OpenAI-compatible API directly.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class CodexLLM(LLMInterface):
    """LLM interface using the local Codex CLI"""

    def __init__(self, model_cfg: Optional[dict] = None):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay

        if not hasattr(logger, "_initialized_codex_models"):
            logger._initialized_codex_models = set()

        if self.model not in logger._initialized_codex_models:
            logger.info(f"Initialized Codex CLI LLM with model: {self.model}")
            logger._initialized_codex_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using Codex CLI with a system message and context"""
        prompt = self._format_prompt(system_message, messages)

        retries = kwargs.get("retries", self.retries) or 0
        retry_delay = kwargs.get("retry_delay", self.retry_delay) or 0
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                return await self._call_codex(prompt, timeout=timeout)
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Codex CLI error on attempt {attempt + 1}/{retries + 1}: {e}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} Codex CLI attempts failed: {e}")
                    raise

    def _format_prompt(self, system_message: str, messages: List[Dict[str, str]]) -> str:
        chunks = [
            "You are being used as an LLM backend for OpenEvolve.",
            "Return only the requested response content. Do not edit files.",
        ]

        if system_message:
            chunks.extend(["", "### SYSTEM", system_message])

        for message in messages:
            role = str(message.get("role", "user")).upper()
            content = message.get("content", "")
            chunks.extend(["", f"### {role}", content])

        return "\n".join(chunks).rstrip() + "\n"

    async def _call_codex(self, prompt: str, timeout: Optional[int]) -> str:
        output_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix="openevolve_codex_", delete=False
            ) as output_file:
                output_path = output_file.name

            cmd = [
                "codex",
                "--ask-for-approval",
                "never",
                "exec",
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--output-last-message",
                output_path,
            ]

            if self.model and self.model != "codex":
                cmd.extend(["--model", self.model])

            cmd.append("-")

            logger.debug(f"Running Codex CLI command: {' '.join(cmd[:-1])} -")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd(),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(prompt.encode("utf-8")), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                raise

            if process.returncode != 0:
                stderr_text = stderr.decode("utf-8", errors="replace").strip()
                stdout_text = stdout.decode("utf-8", errors="replace").strip()
                detail = stderr_text or stdout_text or f"exit code {process.returncode}"
                raise RuntimeError(f"Codex CLI failed: {detail}")

            response = Path(output_path).read_text(encoding="utf-8").strip()
            if not response:
                stdout_text = stdout.decode("utf-8", errors="replace").strip()
                response = stdout_text.strip()

            if not response:
                raise RuntimeError("Codex CLI returned an empty response")

            logger.debug(f"Codex CLI response: {response}")
            return response
        finally:
            if output_path:
                try:
                    Path(output_path).unlink(missing_ok=True)
                except OSError:
                    logger.debug(f"Could not remove temporary Codex output file: {output_path}")
