"""Audited text/multi-image inference for the multi-subgoal experiment.

Credentials stay in memory. No training code or native RoboWits files are used.
Gemini API contract: https://ai.google.dev/api/generate-content
"""
import base64
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from .vlm_reward import generate


class Backend:
    def __init__(self, name, *, socket=None, model='gemini-3-flash-preview',
                 credential_file=None, key_env='RPAD_GEMINI_API_KEY', thinking_level='low', qwen_thinking=False):
        if name not in ('gemini', 'qwen'):
            raise ValueError('Unknown backend')
        self.name, self.socket, self.model = name, socket, model
        if thinking_level not in ('minimal','low','medium','high'):
            raise ValueError('Unsupported Gemini thinking level')
        self.thinking_level = thinking_level
        if type(qwen_thinking) is not bool:
            raise ValueError('qwen_thinking must be boolean')
        self.qwen_thinking = qwen_thinking
        self.key = os.environ.get(key_env)
        if name == 'gemini' and not self.key and credential_file:
            # Read only the requested key; never execute/source shell content.
            for line in Path(credential_file).read_text().splitlines():
                key, separator, value = line.strip().partition('=')
                if separator and key == key_env:
                    self.key = value.strip().strip('\"\'')
                    break
        if name == 'gemini' and not self.key:
            raise RuntimeError(f'Missing credential {key_env}')
        if name == 'qwen' and socket is None:
            raise ValueError('Qwen socket is required')

    def complete(self, prompt, images, out, *, schema=None):
        out = Path(out)
        if self.name == 'qwen':
            # The existing local service belongs to an earlier experiment and
            # may shut down during a long run. Its replacement supervisor owns
            # restart; only retry a missing/reset connection, never invent text.
            connection_errors = []
            for attempt in range(19):
                try:
                    raw = generate(self.socket, prompt, images, out,
                        **({'options':{'enable_thinking':True}} if self.qwen_thinking else {}))
                except (FileNotFoundError, ConnectionRefusedError, ConnectionResetError,
                        BrokenPipeError, json.JSONDecodeError) as exc:
                    connection_errors.append({'attempt':attempt,'type':type(exc).__name__})
                    out.mkdir(parents=True, exist_ok=True)
                    (out/'connection-retries.json').write_text(json.dumps(connection_errors,indent=2))
                    if attempt == 18:
                        raise
                    time.sleep(5)
                else:
                    if self.qwen_thinking and json.loads((out/'response.json').read_text()).get('thinking_enabled') is not True:
                        raise RuntimeError('Local service did not acknowledge requested Qwen thinking mode')
                    return raw
        out.mkdir(parents=True, exist_ok=True)
        config = {'temperature': 1.0, 'maxOutputTokens': 8192,
                  'responseMimeType': 'application/json',
                  'thinkingConfig': {'thinkingLevel': self.thinking_level}}
        if schema is not None:
            config['responseJsonSchema'] = schema
        (out/'request.json').write_text(json.dumps({
            'backend': self.name, 'model': self.model, 'prompt': prompt,
            'images': [str(Path(p).resolve()) for p in images],
            'generationConfig': config}, indent=2))
        parts = [{'text': prompt}]
        for path in images:
            suffix = Path(path).suffix.lower()
            if suffix not in ('.png', '.jpg', '.jpeg'):
                raise ValueError('Only PNG and JPEG inputs are supported')
            parts.append({'inlineData': {
                'mimeType': 'image/png' if suffix == '.png' else 'image/jpeg',
                'data': base64.b64encode(Path(path).read_bytes()).decode()}})
        request = urllib.request.Request(
            f'https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent',
            data=json.dumps({'contents': [{'role': 'user', 'parts': parts}],
                             'generationConfig': config}).encode(),
            headers={'Content-Type': 'application/json', 'x-goog-api-key': self.key})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                body = response.read().decode()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors='replace').replace(self.key, '[REDACTED]')
            (out/'error.json').write_text(json.dumps({'http_status': exc.code, 'body': body}))
            raise RuntimeError(f'Gemini HTTP {exc.code}; see {out}/error.json') from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f'Gemini transport error: {type(exc).__name__}') from None
        payload = json.loads(body.replace(self.key, '[REDACTED]'))
        (out/'provider-response.json').write_text(json.dumps(payload, indent=2))
        candidates = payload.get('candidates', [])
        if len(candidates) != 1 or candidates[0].get('finishReason') != 'STOP':
            raise RuntimeError('Gemini response incomplete or blocked')
        raw = ''.join(p.get('text', '') for p in candidates[0].get('content', {}).get('parts', [])
                      if not p.get('thought', False))
        if not raw.strip():
            raise RuntimeError('Gemini returned no text')
        (out/'response.json').write_text(json.dumps({
            'raw': raw, 'model': payload.get('modelVersion', self.model),
            'seconds': time.monotonic()-started, 'usage': payload.get('usageMetadata')}, indent=2))
        return raw


def strict_json(raw):
    text = raw.rsplit('</think>', 1)[-1].strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON field')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'Non-finite JSON number: {value}')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def validated_complete(backend, prompt, images, out, validator, attempts=3, *, schema=None):
    """Retry malformed answers, retaining every raw response; never repair scores."""
    out = Path(out)
    validation_error = None
    for attempt in range(attempts):
        folder = out/f'attempt-{attempt}'
        reminder = '' if attempt == 0 else (
            '\nYour previous response did not match the requested JSON schema. '
            'Return exactly ONE JSON object with the requested top-level key and fields. '
            'No bare array, markdown, repeated answer, or trailing text. '
            f'Validation error to correct: {validation_error}')
        raw = backend.complete(prompt+reminder, images, folder, **({'schema':schema} if schema else {}))
        try:
            result = validator(raw)
        except (ValueError, TypeError, KeyError) as exc:
            validation_error = str(exc)
            folder.mkdir(parents=True, exist_ok=True)
            (folder/'validation-error.json').write_text(json.dumps({'type':type(exc).__name__, 'error':str(exc)}))
            if attempt+1 == attempts:
                raise ValueError(f'VLM schema failed after {attempts} attempts; see {out}') from exc
        else:
            return result
