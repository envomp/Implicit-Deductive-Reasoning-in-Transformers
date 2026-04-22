SYSTEM_PROMPT = """You are a logical reasoner. Your task is to determine if the query is provable or unprovable based on the given facts and rules.
\nInstructions:
Reason by building a forward-chaining proof. Start with your known facts. 
Apply rules to derive new facts. Repeat until you either derive the query or can no longer apply any new rules.
A rule can ONLY be applied if ALL of its premises are currently present in the known facts.
"""

FINAL_INSTRUCTIONS_COT = """Show your steps clearly.
\nExample step:
- Known facts: 1, 2
- Apply R1 (IF 1 AND 2 THEN 3): Derived new fact 3. Known facts: 1, 2, 3
\nReason step-by-step to reach your conclusion.
If the query is derived, stop and output "Conclusion: provable".
If no more rules apply and query is not derived, output "Conclusion: unprovable".
"""

FINAL_INSTRUCTIONS_DIRECT = """Do not provide any intermediate steps or explanations. 
Based on the facts and rules, output ONLY the final answer in the format "Conclusion: either provable or unprovable"
"""

def construct_prompt(sample, is_cot):
    body = _construct_formal_if_then(sample)
    prompt_lines = [SYSTEM_PROMPT] + body + [FINAL_INSTRUCTIONS_COT if is_cot else FINAL_INSTRUCTIONS_DIRECT]
    return "\n".join(prompt_lines)


def _get_common_sample_data(sample):
    facts = sample.get('facts', [])
    query = sample.get('query', '')
    rules = sample.get('rules', {})
    premises_list = rules.get('premises', [])
    conclusion_list = rules.get('conclusion', [])
    return facts, query, premises_list, conclusion_list


def _construct_formal_if_then(sample):
    facts, query, premises_list, conclusion_list = _get_common_sample_data(sample)

    prompt_lines = []
    prompt_lines.append("Facts (known to be true):")
    prompt_lines.append(f"{', '.join(map(str, facts))}")
    prompt_lines.append("\nRules:")
    for i in range(len(premises_list)):
        prompt_lines.append(f"R{i + 1}: IF {' AND '.join(map(str, premises_list[i]))} THEN {conclusion_list[i]}")
    prompt_lines.append("\nQuery:")
    prompt_lines.append(f"Is the proposition '{query}' true based on the above facts and rules?")
    return prompt_lines
