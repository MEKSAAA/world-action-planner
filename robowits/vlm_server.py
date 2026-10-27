"""Local Qwen3.5 vision-language service; inference only, no training."""
import argparse
import json
import os
from pathlib import Path
import socketserver
import signal
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--max-new-tokens', type=int, default=2048)
    args = p.parse_args()
    if args.socket.exists():
        raise FileExistsError(args.socket)
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, local_files_only=True,
            dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa').eval()
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                request = json.loads(self.rfile.readline(8_000_000))
                content = [{'type': 'image', 'image': Image.open(path).convert('RGB')} for path in request['images']]
                content.append({'type': 'text', 'text': request['prompt']})
                messages = []
                for example in request.get('examples', []):
                    example_content = [{'type':'image','image':Image.open(path).convert('RGB')} for path in example['images']]
                    example_content.append({'type':'text','text':example['prompt']})
                    messages.extend([{'role':'user','content':example_content},
                                     {'role':'assistant','content':example['answer']}])
                messages.append({'role':'user','content':content})
                inputs = processor.apply_chat_template(messages, tokenize=True,
                         add_generation_prompt=True, enable_thinking=False, return_dict=True, return_tensors='pt').to('cuda:0')
                started = time.monotonic()
                with torch.inference_mode():
                    result = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                tokens = result[0, inputs['input_ids'].shape[-1]:]
                response = {'raw': processor.decode(tokens, skip_special_tokens=True), 'model': str(args.model),
                            'seconds': time.monotonic()-started, 'output_tokens': len(tokens),
                            'example_count': len(request.get('examples', [])),
                            'reached_token_limit': len(tokens) >= args.max_new_tokens}
            except Exception as error:
                response = {'error': f'{type(error).__name__}: {error}'}
            self.wfile.write((json.dumps(response)+'\n').encode())
    def terminate(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    try:
        with socketserver.UnixStreamServer(str(args.socket), Handler) as server:
            os.chmod(args.socket, 0o600)
            print('READY', flush=True)
            server.serve_forever()
    finally:
        args.socket.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
