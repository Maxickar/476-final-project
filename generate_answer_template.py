#!/usr/bin/env python3
"""
Generate a placeholder answer file that matches the expected auto-grader format.

Replace the placeholder logic inside `build_answers()` with your own agent loop
before submitting so the ``output`` fields contain your real predictions.

Reads the input questions from cse_476_final_project_test_data.json and writes
an answers JSON file where each entry contains a string under the "output" key.
"""

from __future__ import annotations
import ast, operator as op
import json
from pathlib import Path
from typing import Any, Dict, List
import os, textwrap, re, time
import requests

#added in apikey, apibase, model from the tutorial just replace the key with yours
API_KEY  = os.getenv("OPENAI_API_KEY", "")
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")  
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")  
INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data

             

#based on the final_project_tutorial section i just put it here for testing not sure if we need to change the prompt?
def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 128,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            # try best-effort to surface error text
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}

#first technique: direct prompt enters in the question and gets an answer
def direct(question):
    result = call_model_chat_completions(question["input"])
    answer = result["text"]
    return answer

#second technique self refine basically entering in the question and checking to make sure its good
def refine(question):
    #same thing as direct prompt technique
    result = call_model_chat_completions(question["input"])
    answer = result["text"]

    #calling the api again to ask it to check if the answer matches the question
    checky = call_model_chat_completions("Answer: " + answer + ". Now please verify if this answers the question: " + question["input"] + "\n And optimize the answer to correctly answer the question.")
    return checky
    
# --- PROVIDED: calculator tools from minilab5 of class CSE476, I use this since there is many calculations in the test data, I make some updates to it to still handle the worst case scenarios  ---
SYSTEM_AGENT = """You are a math tool-using agent.
You may do exactly ONE of the following in your reply:
1) CALCULATE: <arithmetic expression>
   - use only numbers, + - * / **, parentheses, and round(x, ndigits)
   - DO NOT include any units/words (e.g., "days", "cups") in the expression, if there is any units/words, try to convert all into the same unit first and then calculate
   - example: CALCULATE: round((3*2.49)*1.07, 2)
2) FINAL: <answer>
Rules:
- Keep track of units if present, but do not include them in the CALCULATE expression. You have to convert all quantities to the same unit before calculating. The unit chosen should be the one that the question ask or the smallest unit if the question does not specify.
- Return ONE final result with unit (if applicable). No other text.
- If there is .0 at the end of a number, you can remove it (e.g., 5.0 -> 5).
Example:
Question: I have been here for 1 year and 2 months. How long have I been here in total? Assume that year is a leap year and I been here since May (May has 31 days)
Correct: 427
"""

def make_first_prompt(question: str) -> str:
    return f"""Question: {question}
If you need arithmetic to get the answer, reply as:
CALCULATE: <expression>
Otherwise reply:
FINAL: <answer>"""

def make_second_prompt(result: str) -> str:
    return f"""The calculation result is: {result}
Now provide the final answer.
Reply exactly as: FINAL: <answer>"""

ACTION_RE = re.compile(r"^\s*(CALCULATE|FINAL)\s*:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)

def parse_action(text: str):
    """
    Returns ("CALCULATE", expr) or ("FINAL", answer); raises ValueError on bad format.
    """
    m = ACTION_RE.match(text.strip())
    if not m:
        raise ValueError(f"Unrecognized action format: {text!r}")
    action = m.group(1).upper()
    payload = m.group(2).strip()
    return action, payload

#Function that evaluates arithmetic expressions.
ALLOWED_BINOPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod}
ALLOWED_UNOPS  = {ast.UAdd: op.pos, ast.USub: op.neg}
def safe_eval(expr: str):
    """
    Evaluates a tiny arithmetic language: numbers, + - * / ** % parentheses, round(x, ndigits).
    Converts '^' to '**'. Rejects anything else.
    """
    expr = expr.replace("^", "**")
    if len(expr) > 200:
        raise ValueError("Expression too long.")
    node = ast.parse(expr, mode="eval")
    def ev(n):
        if isinstance(n, ast.Expression):  return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.UnaryOp) and type(n.op) in ALLOWED_UNOPS:        return ALLOWED_UNOPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.BinOp) and type(n.op) in ALLOWED_BINOPS:         return ALLOWED_BINOPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "round":
            args = [ev(a) for a in n.args]
            return round(*args)
        if isinstance(n, ast.Tuple):  # allow round(x,2) with comma
            return tuple(ev(elt) for elt in n.elts)
        raise ValueError(f"Disallowed expression: {ast.dump(n, include_attributes=False)}")
    return ev(node)

# seventh technique: tool-augmented prompting, we will use the model to call an external tool (e.g. a calculator) to help answer the question
def tool_augmented(question, verbose: bool = True): #verbose is for debugging purposes

    r1 = call_model_chat_completions(make_first_prompt(question["input"]), system=SYSTEM_AGENT)

    if not r1["ok"] or not r1["text"]:
        return ""
    
    if verbose: print("LLM ->", r1["text"])

    try:
        action, payload = parse_action(r1["text"])
    except Exception:
        return r1["text"].strip()  # fallback raw answer if parsing fails
    
    if action == "FINAL":
        return payload.strip()
    
    if action == "CALCULATE":
        try:
            calc_value = safe_eval(payload)
        except Exception:
            return r1["text"].strip()  # fallback raw answer if calculation fails
        if verbose: print("CALC =", calc_value)
        # send results back to model for final answer
        rN = call_model_chat_completions(make_second_prompt(calc_value), system=SYSTEM_AGENT)
        if not rN["ok"] or not rN["text"]:
            return str(calc_value)  # fallback to just returning the calculation result if model fails on second step
        if verbose: print("LLM →", rN["text"])
        try:
            action2, payload2 = parse_action(rN["text"])
            if action2 == "FINAL":
                return payload2.strip()
        except Exception:
            pass
        #Fallback
        return str(calc_value)
    
    #Final fallback if action is unrecognized
    return r1["text"].strip()

#this is how we put in the techniques just replce the method name in direct(question) to another technique
def build_answers(questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    answers = []
    for idx, question in enumerate(questions, start=1):

        #change the method to one of the technques
        answer = refine(question)
        answers.append({"output": answer})
        print(f"{idx} / {len(questions)} done")
    return answers


def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars). Please make sure your answer does not include any intermediate results."
            )


def main() -> None:
    questions = load_questions(INPUT_PATH)
    answers = build_answers(questions)

    with OUTPUT_PATH.open("w") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)

    with OUTPUT_PATH.open("r") as fp:
        saved_answers = json.load(fp)
    validate_results(questions, saved_answers)
    print(
        f"Wrote {len(answers)} answers to {OUTPUT_PATH} "
        "and validated format successfully."
    )


if __name__ == "__main__":
    main()

