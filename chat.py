import argparse
import sys
import torch
import torch.nn.functional as F
from raplib.model import ModelConfig, PrimitiveManifoldLM
from raplib.data import ids_to_bytes

def parse_args():
    parser = argparse.ArgumentParser(description="Chat with RAP model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_len", type=int, default=256, help="Max response length")
    return parser.parse_args()

def load_model(ckpt_path, device):
    print(f"Loading model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    if "cfg" in ckpt:
        cfg_dict = ckpt["cfg"]
        valid_keys = ModelConfig.__annotations__.keys()
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        cfg = ModelConfig(**filtered_cfg)
    else:
        cfg = ModelConfig()

    model = PrimitiveManifoldLM(cfg).to(device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg

def generate_response(model, prompt_ids, state, args):
    device = next(model.parameters()).device
    curr_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    # Process prompt
    with torch.no_grad():
        out = model(curr_ids, state)
        logits_seq, state = out[0], out[1]
        next_logits = logits_seq[:, -1, :]
        
        generated_ids = []
        sys.stdout.write("AI: ")
        sys.stdout.flush()
        
        for _ in range(args.max_len):
            logits = next_logits / args.temperature
            if args.top_k > 0:
                v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            token_int = next_token.item()
            
            # Stop if newline (simple chat format)
            if token_int == 10: # \n
                break
                
            generated_ids.append(token_int)
            
            # Streaming print (best effort)
            try:
                # Try to decode the single byte
                char = bytes([token_int]).decode("utf-8")
                sys.stdout.write(char)
                sys.stdout.flush()
            except:
                pass # Wait for valid utf-8 sequence or just ignore
            
            # Forward step
            out = model(next_token, state)
            logits_seq, state = out[0], out[1]
            next_logits = logits_seq[:, -1, :]
            
        sys.stdout.write("\n")
        return generated_ids, state

def main():
    args = parse_args()
    device = torch.device(args.device)
    model, cfg = load_model(args.ckpt, device)
    
    print("\n--- RAP Chat (Ctrl+C to exit) ---")
    print("Format: 'Question: <input>\\n\\nAnswer: <output>'")
    
    # Initialize empty state
    # In a real recurrent setting, we might want to keep history.
    # But for this simple model trained on short sequences, 
    # resetting state or keeping a sliding window might be better.
    # Let's try continuous state for the session.
    state = model.init_state(batch_size=1, device=device)
    
    while True:
        try:
            user_input = input("\nYou: ")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
        if not user_input.strip():
            continue
            
        # Format as per training data (roughly)
        # Assuming training data was just raw text, but let's frame it as Q/A if helpful
        # Or just append.
        # Let's just feed raw input + newline.
        
        prompt_text = f"Question: {user_input}\nAnswer:"
        prompt_bytes = prompt_text.encode("utf-8")
        prompt_ids = [int(b) for b in prompt_bytes]
        
        # Reset state for each turn to avoid pollution from previous bad generations?
        # Or keep it? The model has limited memory (d_state=512).
        # Let's reset for now to ensure clean responses.
        state = model.init_state(batch_size=1, device=device)
        
        _, _ = generate_response(model, prompt_ids, state, args)

if __name__ == "__main__":
    main()
