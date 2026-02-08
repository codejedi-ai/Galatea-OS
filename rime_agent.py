import logging
import random
import os
import json
import argparse
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    tts,
    metrics,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli
)
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import (
    openai,
    noise_cancellation,
    rime,
    silero,
)
from livekit.agents.tokenize import tokenizer

from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agent_configs import VOICE_CONFIGS

load_dotenv()
logger = logging.getLogger("voice-agent")

VOICE_NAMES = ["celeste"]
# randomly select a voice from the list
VOICE = random.choice(VOICE_NAMES)

# Global config loaded from JSON file (set when --config is used)
LOADED_CONFIG = None

def load_config_from_file(config_path: str) -> dict:
    """Load agent configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    logger.info(f"Loaded config from {config_path}: {config.get('name', 'unknown')}")
    return config

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

class RimeAssistant(Agent):
    def __init__(self, prompt: str = None) -> None:
        if prompt is None:
            prompt = VOICE_CONFIGS[VOICE]["llm_prompt"]
        super().__init__(instructions=prompt)


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()

    # Determine which configuration to use
    if LOADED_CONFIG:
        voice_name = LOADED_CONFIG.get("name", "custom")
        logger.info(f"Running Rime voice agent with loaded config: {voice_name} for participant {participant.identity}")
        
        # Extract TTS options from loaded config
        tts_provider = LOADED_CONFIG.get("tts_type", "rime")
        voice_options = LOADED_CONFIG.get("voice_options", {})
        
        # Use Rime TTS with configuration from JSON
        rime_tts = rime.TTS(
            model=voice_options.get("model", "arcana"),
            speaker=voice_options.get("speaker", "celeste"),
            speed_alpha=voice_options.get("speed_alpha", 1.5),
            reduce_latency=voice_options.get("reduce_latency", True),
            max_tokens=voice_options.get("max_tokens", 3400),
        )
        
        llm_prompt = LOADED_CONFIG.get("personality_prompt", "You are a helpful assistant.")
        intro_phrase = LOADED_CONFIG.get("greeting", {}).get("intro_phrase", "Hello!")
    else:
        voice_name = VOICE
        logger.info(f"Running Rime voice agent for voice config {voice_name} and participant {participant.identity}")
        
        rime_tts = rime.TTS(
            **VOICE_CONFIGS[VOICE]["tts_options"]
        )
        if VOICE_CONFIGS[VOICE].get("sentence_tokenizer"):
            sentence_tokenizer = VOICE_CONFIGS[VOICE].get("sentence_tokenizer")
            if not isinstance(sentence_tokenizer, tokenizer.SentenceTokenizer):
                raise TypeError(
                    f"Expected sentence_tokenizer to be an instance of tokenizer.SentenceTokenizer, got {type(sentence_tokenizer)}"
                )
            rime_tts = tts.StreamAdapter(tts=rime_tts, sentence_tokenizer=sentence_tokenizer)
        
        llm_prompt = VOICE_CONFIGS[VOICE]["llm_prompt"]
        intro_phrase = VOICE_CONFIGS[VOICE]["intro_phrase"]

    session = AgentSession(
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=rime_tts,
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel()
    )
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        room=ctx.room,
        agent=RimeAssistant(prompt=llm_prompt),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
        room_output_options=RoomOutputOptions(audio_enabled=True),
    )

    await session.say(intro_phrase)

def _parse_config_and_run():
    """Parse --config from argv, set LOADED_CONFIG, then run the app."""
    import sys
    config_file = None
    if "--config" in sys.argv:
        config_idx = sys.argv.index("--config")
        if config_idx + 1 < len(sys.argv):
            config_file = sys.argv[config_idx + 1]
            sys.argv.pop(config_idx)
            sys.argv.pop(config_idx)
    global LOADED_CONFIG
    if config_file:
        LOADED_CONFIG = load_config_from_file(config_file)
        logger.info(f"Using config from {config_file}")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )

if __name__ == "__main__":
    _parse_config_and_run()
