import argparse
import sys
import torch
import torch.nn.functional as F
from raplib.model import ModelConfig, PrimitiveManifoldLM
from raplib.data import ids_to_bytes, strip_pos_tag_bytes

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using RAP model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="The", help="Text prompt to start generation")
    parser.add_argument("--length", type=int, default=100, help="Number of bytes to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading model from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device)
    
    # Load Config
    if "cfg" in ckpt:
        cfg_dict = ckpt["cfg"]
        # Filter out any keys that might not be in the current ModelConfig definition (forward compatibility)
        valid_keys = ModelConfig.__annotations__.keys()
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        cfg = ModelConfig(**filtered_cfg)
    else:
        # Fallback for older checkpoints or raw state dicts (assumes default config)
        print("Warning: No config found in checkpoint, using defaults.")
        cfg = ModelConfig()

    print(f"Model Config: {cfg}")

    # Initialize Model
    model = PrimitiveManifoldLM(cfg).to(device)
    
    # Load Weights
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # Prepare Prompt
    prompt_bytes = args.prompt.encode("utf-8")
    prompt_ids = [int(b) for b in prompt_bytes]
    curr_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device) # (1, T)

    # Init State
    state = model.init_state(batch_size=1, device=device)

    # Prefill (process prompt to update state)
    # We can process the whole prompt at once to get the final state
    # But we need to be careful: the model returns sequence of logits.
    # The last logit is the prediction for the NEXT token after the last prompt token.
    
    with torch.no_grad():
        # Run the model on the prompt
        out = model(curr_ids, state)
        logits_seq, state = out[0], out[1]
        # The last logit in the sequence is for the next token prediction
        next_logits = logits_seq[:, -1, :] 

        generated_ids = list(prompt_ids)
        
        print(f"\nPrompt: '{args.prompt}'")
        sys.stdout.write(args.prompt)
        sys.stdout.flush()

        for _ in range(args.length):
            # Sampling
            logits = next_logits / args.temperature
            
            # Top-K
            if args.top_k > 0:
                v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            token_int = next_token.item()
            generated_ids.append(token_int)

            # Print the new byte (if it's a valid char, otherwise placeholder)
            try:
                # We decode just this byte to see if it prints, 
                # but for correct utf-8 we should really decode the whole buffer.
                # For streaming output, we'll try best effort or just accumulate.
                # Simple approach: decode the growing buffer and print the new part if possible.
                # Actually, simpler: just print the raw byte assuming latin-1 for debug or try utf-8
                # Let's rely on the final decode for correctness and just print '.' for partials?
                # No, let's try to decode the last few bytes.
                
                # We'll just print the final result at the end to avoid broken utf-8 chars in terminal
                pass 
            except:
                pass

            # Update for next step
            next_input = next_token # (1, 1)
            
            # Forward pass just for the new token
            # We must pass the state from the previous step
            # In PrimitiveManifoldLM.forward:
            # logits, state = self.step(x_emb, state)
            # So passing (1,1) input works fine and updates state.
            
            # Note: PrimitiveManifoldLM.forward expects (batch, time)
            # It returns (batch, time, vocab), state
            
            out = model(next_input, state)
            logits_seq, state = out[0], out[1]
            next_logits = logits_seq[:, -1, :]

    # Final Decode
    print("\n\n--- Generated Output ---")
    try:
        out_b = ids_to_bytes(generated_ids)
        out_b = strip_pos_tag_bytes(out_b)
        decoded_text = out_b.decode("utf-8", errors="replace")
        print(decoded_text)
    except Exception as e:
        print(f"Error decoding: {e}")
        print(f"Raw bytes: {ids_to_bytes(generated_ids)}")

if __name__ == "__main__":
    main()
