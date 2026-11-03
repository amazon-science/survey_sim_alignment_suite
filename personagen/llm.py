# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""AWS Bedrock LLM client with retry logic and parallel execution."""

from __future__ import annotations
import json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import boto3
from botocore.config import Config

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_IDS: Dict[str, str] = {
    "claude3.7": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "claude4.0": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "llama3.3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
    "nova-pro": "us.amazon.nova-pro-v1:0",
}

_CFG = Config(read_timeout=900, connect_timeout=900,
              retries={"max_attempts": 10, "mode": "standard"})

# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def _format_input(model: str, system: str, user: str,
                  max_tokens: int, temperature: float) -> dict:
    if model.startswith("claude"):
        return {"anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens, "system": system,
                "messages": [{"role": "user", "content": user}],
                "temperature": temperature}
    if model.startswith("llama"):
        prompt = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                  f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                  f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        return {"prompt": prompt, "max_gen_len": max_tokens,
                "temperature": temperature, "top_p": 1.0}
    if model.startswith("nova"):
        body: dict = {"messages": [{"role": "user", "content": [{"text": user}]}],
                      "inferenceConfig": {"max_new_tokens": max_tokens,
                                          "temperature": temperature, "topP": 1.0}}
        if system:
            body["system"] = [{"text": system}]
        return body
    raise ValueError(f"Unsupported model: {model}")


def _extract_text(body: dict, model: str) -> str:
    if model.startswith("claude"):
        return body["content"][0]["text"]
    if model.startswith("llama"):
        return body["generation"]
    if model.startswith("nova"):
        return body["output"]["message"]["content"][0]["text"]
    raise ValueError(f"Unsupported model: {model}")


# ---------------------------------------------------------------------------
# Core call
# ---------------------------------------------------------------------------

def call_bedrock(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    region: str = "us-east-1",
    profile: str | None = None,
    max_retries: int = 6,
) -> str:
    """Call Bedrock and return the response text. Retries on throttle."""
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    client = session.client("bedrock-runtime", region_name=region, config=_CFG)
    body = _format_input(model, system_prompt, user_prompt, max_tokens, temperature)
    bedrock_id = MODEL_IDS.get(model, model)

    for attempt in range(max_retries):
        try:
            resp = client.invoke_model(modelId=bedrock_id, body=json.dumps(body))
            return _extract_text(json.loads(resp["body"].read()), model)
        except Exception as e:
            print(f"  [attempt {attempt+1}/{max_retries}] {type(e).__name__}: {e}")
            if attempt == max_retries - 1:
                raise
            wait = 120 if "Throttling" in str(e) else 60
            time.sleep(wait)
    return ""


# ---------------------------------------------------------------------------
# Parallel helper
# ---------------------------------------------------------------------------

def call_bedrock_parallel(
    prompts: List[Tuple[str, str]],
    model: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    region: str = "us-east-1",
    profile: str | None = None,
    workers: int = 8,
    on_result=None,
) -> List[str]:
    """Call Bedrock in parallel for a list of (system, user) prompt pairs.

    Returns a list of response texts in the same order as *prompts*.
    on_result: optional callback(idx, text) called as each result arrives.
    """
    results: Dict[int, str] = {}

    def _task(idx: int, sys_p: str, usr_p: str) -> Tuple[int, str]:
        txt = call_bedrock(model, sys_p, usr_p, max_tokens=max_tokens,
                           temperature=temperature, region=region, profile=profile)
        return idx, txt

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_task, i, s, u): i for i, (s, u) in enumerate(prompts)}
        from tqdm import tqdm
        for fut in tqdm(as_completed(futures), total=len(prompts), desc="Bedrock calls"):
            idx, txt = fut.result()
            results[idx] = txt
            if on_result:
                on_result(idx, txt)

    return [results[i] for i in range(len(prompts))]


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def call_bedrock_batch(
    prompts: List[Tuple[str, str]],
    model: str,
    *,
    s3_bucket: str,
    s3_prefix: str = "personagen/batch",
    max_tokens: int = 256,
    temperature: float = 0.0,
    region: str = "us-east-1",
    profile: str | None = None,
    role_arn: str,
    poll_interval: int = 30,
) -> List[str]:
    """Submit Bedrock batch inference job, poll, return results in order."""
    import uuid
    from datetime import datetime

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    s3 = session.client("s3", region_name=region)
    bedrock = session.client("bedrock", region_name=region)
    bedrock_id = MODEL_IDS.get(model, model)

    job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]

    # Build JSONL records
    MAX_RECORDS_PER_JOB = 90000
    all_lines = []
    for i, (sys_p, usr_p) in enumerate(prompts):
        record = {
            "recordId": f"{i:06d}",
            "modelInput": _format_input(model, sys_p, usr_p, max_tokens, temperature),
        }
        all_lines.append(json.dumps(record, ensure_ascii=False))

    # Split into chunks if needed, submit multiple jobs
    chunks = [all_lines[i:i + MAX_RECORDS_PER_JOB] for i in range(0, len(all_lines), MAX_RECORDS_PER_JOB)]
    job_arns = []

    for chunk_idx, chunk in enumerate(chunks):
        chunk_id = f"{job_id}-{chunk_idx}" if len(chunks) > 1 else job_id
        input_key = f"{s3_prefix}/{chunk_id}/input.jsonl"
        s3.put_object(Bucket=s3_bucket, Key=input_key, Body="\n".join(chunk).encode("utf-8"))

        resp = bedrock.create_model_invocation_job(
            jobName=f"personagen-{chunk_id}",
            modelId=bedrock_id,
            roleArn=role_arn,
            inputDataConfig={"s3InputDataConfig": {"s3Uri": f"s3://{s3_bucket}/{input_key}"}},
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": f"s3://{s3_bucket}/{s3_prefix}/{chunk_id}/output/"}},
        )
        job_arns.append(resp["jobArn"])

    print(f"📤 Uploaded {len(prompts)} records in {len(chunks)} job(s)")
    print(f"🚀 Batch job(s): {job_arns}")

    # Poll all jobs
    pending = list(job_arns)
    while pending:
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            bedrock = session.client("bedrock", region_name=region)
            still_pending = []
            for arn in pending:
                status = bedrock.get_model_invocation_job(jobIdentifier=arn)
                state = status["status"]
                if state == "Completed":
                    print(f"✅ {arn.split('/')[-1]} complete")
                elif state in ("Failed", "Stopped"):
                    raise RuntimeError(f"Batch {state}: {status.get('message', '')}")
                else:
                    still_pending.append(arn)
            pending = still_pending
            if pending:
                time.sleep(poll_interval)
        except RuntimeError:
            raise
        except Exception as e:
            print(f"  ⚠️  Poll error: {e}. Retrying in {poll_interval}s...")
            time.sleep(poll_interval)

    # Download results from all chunks
    results: Dict[int, str] = {}
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    s3 = session.client("s3", region_name=region)
    for chunk_idx in range(len(chunks)):
        chunk_id = f"{job_id}-{chunk_idx}" if len(chunks) > 1 else job_id
        out_prefix = f"{s3_prefix}/{chunk_id}/output/"
        objs = s3.list_objects_v2(Bucket=s3_bucket, Prefix=out_prefix)
        for obj in objs.get("Contents", []):
            if not obj["Key"].endswith(".jsonl.out"):
                continue
            body = s3.get_object(Bucket=s3_bucket, Key=obj["Key"])["Body"].read().decode("utf-8")
            for line in body.strip().split("\n"):
                if not line:
                    continue
                rec = json.loads(line)
                idx = int(rec["recordId"])
                output = rec.get("modelOutput", {})
                try:
                    results[idx] = _extract_text(output, model) if output else ""
                except (KeyError, IndexError):
                    results[idx] = ""

    return [results.get(i, "") for i in range(len(prompts))]
