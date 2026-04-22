null_e = 0
fact_e = 1
query_e = 2
rule_e = 3
rulestart_e = 4
ruleend_e = 5
provable_e = 6
unknown_e = 7
task_e = 8

TYPE_NAME_MAP = {
    fact_e: "Fact",
    query_e: "Query",
    rule_e: "Rule",
    rulestart_e: "Rule start",
    ruleend_e: "Rule end",
    task_e: "Task"
}


##### these work on rule format of {"premise": head, "conclusion": tail} #####

def parse_problem(input_ids_list, type_ids_list):
    facts = set()
    rules = []
    query = None

    current_premise = []
    in_rule_definition = False
    for i, types in enumerate(type_ids_list):
        token_id = input_ids_list[i]

        if task_e in types:
            break

        if fact_e in types:
            facts.add(token_id)

        if query_e in types:
            query = token_id

        if rulestart_e in types and not in_rule_definition:
            in_rule_definition = True

        if in_rule_definition:
            if rulestart_e in types:
                current_premise.append(token_id)
            if ruleend_e in types:
                rules.append({'premise': set(current_premise), 'conclusion': token_id})
                current_premise = []
                in_rule_definition = False
    return facts, rules, query


def solve_tokenized(input_ids, type_ids, layer_by_layer=False):
    input_ids_list = input_ids.squeeze(0).cpu().tolist()
    type_ids_list = type_ids.squeeze(0).cpu().tolist()
    facts, rules, query = parse_problem(input_ids_list, type_ids_list)
    history = [facts.copy()]
    if not rules:
        return history if layer_by_layer else facts

    current_facts = facts.copy()
    while True:
        new_fact_found_this_iteration = False
        next_step_facts = current_facts.copy()
        for rule in rules:
            if rule['premise'].issubset(current_facts) and rule['conclusion'] not in current_facts:
                next_step_facts.add(rule['conclusion'])
                new_fact_found_this_iteration = True

        if new_fact_found_this_iteration:
            current_facts = next_step_facts
            history.append(current_facts.copy())
        else:
            break
    return history if layer_by_layer else current_facts


def calculate_depths(initial_facts, rules):
    depths = {fact: 0 for fact in initial_facts}
    provable_facts = initial_facts.copy()

    while True:
        newly_proven_this_iteration = set()
        for rule in rules:
            if rule['conclusion'] not in provable_facts:
                if rule['premise'].issubset(provable_facts):
                    premise_max_depth = max(depths[p] for p in rule['premise'])
                    depths[rule['conclusion']] = premise_max_depth + 1
                    newly_proven_this_iteration.add(rule['conclusion'])

        if not newly_proven_this_iteration:
            break
        provable_facts.update(newly_proven_this_iteration)

    return depths
