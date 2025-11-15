import json
from datetime import datetime

INSTR = (
"You are an AML rule engine simulator. "
"Given the facts (JSON), output a JSON with fields: "
'{"aml_decision":"CLEAR|REVIEW|SAR|BLOCK","reasons":[...],"escalation_level":0-3}. '
"Respond with JSON only."
)

def json_datetime_serializer(obj):
    """JSON serializer for objects not serializable by default, like datetime."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def row_to_example(row):
    facts = row["facts"]
    decision = row["decision"]
    prompt = f"{INSTR}\n\nFacts (JSON):\n{json.dumps(facts, ensure_ascii=False, default=json_datetime_serializer)}"
    target = json.dumps({
        "aml_decision": decision["aml_decision"],
        "reasons": decision.get("reasons", []),
        "escalation_level": decision.get("escalation_level", 0)
    }, ensure_ascii=False)
    return {"prompt": prompt, "response": target}
