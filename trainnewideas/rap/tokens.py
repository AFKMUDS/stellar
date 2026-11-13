"""All 256 byte values remain available; structure lives above the byte range."""
BOS, EOS, SYSTEM, USER, ASSISTANT, TURN_END = range(256, 262)
VOCAB_SIZE = 262
ROLES = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}
SPECIAL_NAMES = {BOS: "<BOS>", EOS: "<EOS>", SYSTEM: "<SYSTEM>",
                 USER: "<USER>", ASSISTANT: "<ASSISTANT>", TURN_END: "<TURN_END>"}

def encode_text(text):
    return list(text.encode("utf-8"))

def serialize_record(messages=None, text=None):
    ids, mask = [BOS], [0]
    if text is not None:
        raw = encode_text(text)
        ids += raw + [EOS]
        mask += [1] * (len(raw) + 1)
        return ids, mask
    for message in messages or []:
        role = message["role"]
        ids.append(ROLES[role]); mask.append(0)
        raw = encode_text(message["content"])
        ids += raw; mask += [int(role == "assistant")] * len(raw)
        ids.append(EOS if role == "assistant" else TURN_END)
        mask.append(int(role == "assistant"))
    return ids, mask

