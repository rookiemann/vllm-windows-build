"""
vLLM OpenAI-compatible server launcher for Windows.

vLLM's built-in API server uses AsyncMPClient with ZMQ multiprocess,
which doesn't work on Windows (no fork, IPC sockets, etc.).

This launcher creates a lightweight FastAPI server wrapping the
synchronous LLM class (which uses InprocClient on Windows) and
serves the OpenAI-compatible chat/completions endpoint with SSE streaming.
"""

import sys
import types
import os
import asyncio
import argparse
import json
import time
import logging
import uuid

# Stub uvloop on Windows before any vLLM imports
if sys.platform == "win32":
    uvloop = types.ModuleType("uvloop")
    uvloop.run = asyncio.run
    sys.modules["uvloop"] = uvloop

from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vllm_launcher")


def create_app(llm: LLM, model_name: str):
    """Create FastAPI app with OpenAI-compatible endpoints."""
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse, JSONResponse

    app = FastAPI()
    engine = llm.llm_engine

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok"})

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse({
            "object": "list",
            "data": [{
                "id": model_name,
                "object": "model",
                "owned_by": "vllm",
            }]
        })

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()

        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 1024)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.95)
        stream = body.get("stream", False)
        stop = body.get("stop")
        top_k = body.get("top_k", -1)
        repetition_penalty = body.get("repetition_penalty", 1.0)
        presence_penalty = body.get("presence_penalty", 0.0)
        frequency_penalty = body.get("frequency_penalty", 0.0)

        # Build prompt from messages using tokenizer's chat template
        tokenizer = llm.get_tokenizer()
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: simple concatenation
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content if p.get("type") == "text"
                    )
                parts.append(f"{role}: {content}")
            parts.append("assistant:")
            prompt = "\n".join(parts)

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop=stop,
        )

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if not stream:
            # Non-streaming: use LLM.generate()
            outputs = llm.generate([prompt], params)
            text = outputs[0].outputs[0].text
            tokens = len(outputs[0].outputs[0].token_ids)
            return JSONResponse({
                "id": request_id,
                "object": "chat.completion",
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "completion_tokens": tokens,
                    "prompt_tokens": len(outputs[0].prompt_token_ids),
                    "total_tokens": len(outputs[0].prompt_token_ids) + tokens,
                },
            })

        # Streaming: use engine.add_request() + engine.step()
        async def stream_response():
            rid = request_id
            engine.add_request(rid, prompt, params)

            prev_len = 0
            while True:
                step_outputs = engine.step()
                for output in step_outputs:
                    if output.request_id != rid:
                        continue
                    text = output.outputs[0].text
                    new_text = text[prev_len:]
                    prev_len = len(text)

                    if new_text:
                        chunk = {
                            "id": rid,
                            "object": "chat.completion.chunk",
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": new_text},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    if output.finished:
                        finish_chunk = {
                            "id": rid,
                            "object": "chat.completion.chunk",
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }],
                        }
                        yield f"data: {json.dumps(finish_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                # Small yield to avoid blocking
                await asyncio.sleep(0)

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="vLLM Windows Server")
    parser.add_argument("--model", required=True, help="Model path or HF ID")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    logger.info(f"Loading model: {args.model}")
    start = time.time()

    llm_kwargs = {
        "model": args.model,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "enforce_eager": args.enforce_eager,
    }
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)
    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s")

    model_name = os.path.basename(args.model)
    app = create_app(llm, model_name)

    import uvicorn
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                loop="asyncio")


if __name__ == "__main__":
    main()
