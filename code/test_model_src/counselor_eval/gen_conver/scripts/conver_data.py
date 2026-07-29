import json
import os
from pathlib import Path


RESPONSE_MODEL = "test"


# INPUT_PATH = Path("/WX24061/lzy/test_model/counselor_eval/gen_conver/output_data/novice/novice_1.jsonl")
INPUT_PATH = Path("/WX24061/lzy/test_model/counselor_eval/gen_conver/output_data/test/test_5.jsonl")
OUTPUT_DIR = Path("/WX24061/lzy/test_model/counselor_eval/gen_conver/converted_data/test_5")


def normalize_role(role: str) -> str:
    if role == "Therapist":
        return "Counselor"
    return role


def convert_record(record: dict, index: int) -> dict:
    history = record.get("history", [])
    session_dialogue = []

    for turn in history:
        role = normalize_role(turn.get("role", ""))
        text = turn.get("content", "")
        session_dialogue.append({
            "role": role,
            "text": text
        })

    client_id = f"simpsydial_{index}"

    converted = {
        "client_id": client_id,
        "client_info": {
            "topic": "",
            "main_problem": "",
            "core_demands": "",
            "responder_model": RESPONSE_MODEL,
            "source_dataset": "",
            "metadata": {
                "system_prompt": "",
                "total_messages": len(session_dialogue)
            }
        },
        "sessions": [
            {
                "session_number": 1,
                "session_dialogue": session_dialogue
            }
        ]
    }
    return converted


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            converted = convert_record(record, index)

            output_path = OUTPUT_DIR / f"simpsydial_{index}.json"
            with output_path.open("w", encoding="utf-8") as out_f:
                json.dump(converted, out_f, ensure_ascii=False, indent=2)

            print(f"saved: {output_path}")


if __name__ == "__main__":
    main()