import copy
import random
from concurrent.futures.process import ProcessPoolExecutor


##### works on rule format of [[head], tail] #####

def is_reachable(facts, rules, query):
    reachable = set(facts)
    changed = True
    while changed:
        changed = False
        for premises, conclusion in rules:
            if all(premise in reachable for premise in premises) and conclusion not in reachable:
                reachable.add(conclusion)
                changed = True
    return 1 if query in reachable else 0


def is_reachable_in_budget(facts, rules, preds, depth):
    reachable = set(facts)
    reachable_at_depth = {fact: 0 for fact in facts}

    if depth == 0:
        return set(p for p in preds if p in reachable)

    for current_depth in range(1, depth + 1):
        changed = False
        for rule in rules:
            premises, conclusion = rule
            if all(premise in reachable for premise in premises) and conclusion not in reachable_at_depth:
                reachable.add(conclusion)
                reachable_at_depth[conclusion] = current_depth
                changed = True

        if not changed:
            break

    reachable_preds = set(p for p in preds if p in reachable)
    return reachable_preds


def calculate_depths(initial_facts, rules):
    depths = {fact: 0 for fact in initial_facts}
    provable_facts = initial_facts.copy()

    while True:
        newly_proven_this_iteration = set()
        for rule in rules:
            if rule[1] not in provable_facts:
                if set(rule[0]).issubset(provable_facts):
                    premise_max_depth = 0 if not rule[0] else max(depths[p] for p in rule[0])
                    depths[rule[1]] = premise_max_depth + 1
                    newly_proven_this_iteration.add(rule[1])

        if not newly_proven_this_iteration:
            break
        provable_facts.update(newly_proven_this_iteration)

    return depths if depths else {-1: 0}


def is_logically_consistent(facts, rules, query, label):
    """
    Checks if the given example is logically consistent.

    Args:
        facts: List of fact predicates (strings).
        rules: List of rules ( [ [head predicates], tail predicate ] ).
        query: The query predicate (string).
        label: The expected label (0 or 1).

    Returns:
        True if consistent, False otherwise.
    """
    derivable = set(facts)

    changed = True
    while changed:
        changed = False
        for head, tail in rules:
            if all(h in derivable for h in head) and tail not in derivable:
                derivable.add(tail)
                changed = True

    if label == 1:
        return query in derivable
    else:
        return query not in derivable


def solve(facts, rules, query, break_early=True, include_copy=False, eager=True):
    reached = list(facts)
    smooth_reached = [[x] for x in facts]
    changed = query not in list(facts) or not break_early

    while changed:
        changed = False
        next_nodes = []
        for premises, conclusion in rules:
            if all(premise in reached for premise in premises) and conclusion not in reached:
                next_nodes.append(conclusion)
                changed = True
        if changed:
            if eager:
                next_node = next_nodes[0]
            else:
                next_node = random.choice(next_nodes)
            reached.append(next_node)
            smooth_reached.append(next_nodes)
            if next_node == query and break_early:
                break

    labels = reached if include_copy else reached[len(facts):]
    smooth_labels = smooth_reached if include_copy else smooth_reached[len(facts):]
    return labels, smooth_labels


### Modification


def alter_query(new_item, new_label: int):
    new_item["label"] = new_label
    for new_query in new_item["preds"]:
        new_item["query"] = new_query
        if is_logically_consistent(new_item["facts"], new_item["rules"], new_item["query"], new_item["label"]):
            return new_item
    return None


def _is_duplicate_rule(new_rule, existing_rules):
    """Checks if a new rule is a logical duplicate of an existing one."""
    new_head_sorted = sorted(new_rule[0])
    new_body = new_rule[1:]

    for rule in existing_rules:
        # Basic validation
        if not rule or not isinstance(rule[0], list) or len(rule) != len(new_rule):
            continue
        existing_head_sorted = sorted(rule[0])
        existing_body = rule[1:]
        if new_head_sorted == existing_head_sorted and new_body == existing_body:
            return True
    return False


def _add_balancing_rule(item, removed_rule):
    if random.random() < 0.2:
        return

    rule_tail = removed_rule[1]
    available_preds = set(item["preds"]) - {rule_tail}

    provable_depths = calculate_depths(set(item["facts"]), item["rules"])
    provable_candidates = {p: d for p, d in provable_depths.items() if p in available_preds}
    new_head_preds = []
    guaranteed_deep_pred = None

    if provable_candidates:
        max_depth = max(provable_candidates.values())
        highest_depth_preds = [p for p, d in provable_candidates.items() if d == max_depth]
        guaranteed_deep_pred = random.choice(highest_depth_preds)
        new_head_preds.append(guaranteed_deep_pred)

    total_head_count = random.randint(1, min(3, len(available_preds)))
    num_to_sample_more = total_head_count - len(new_head_preds)
    if num_to_sample_more > 0:
        other_candidates_pool = list(available_preds - {guaranteed_deep_pred})
        if other_candidates_pool:
            additional_preds = random.sample(other_candidates_pool, min(len(other_candidates_pool), num_to_sample_more))
            new_head_preds.extend(additional_preds)

    if new_head_preds:
        new_distractor_rule = [new_head_preds, rule_tail]
        if not _is_duplicate_rule(new_distractor_rule, item["rules"]):
            item["rules"].append(new_distractor_rule)


def _add_balancing_fact(item, removed_fact):
    if random.random() < 0.1:
        return

    all_preds = set(item["preds"])
    current_facts_set = set(item["facts"])
    query_pred = {item["query"]}
    candidates = all_preds - current_facts_set - query_pred
    scored_candidates = []

    for potential_fact in candidates:
        temp_facts = current_facts_set | {potential_fact}
        depths = calculate_depths(temp_facts, item["rules"])
        max_depth = max(depths.values()) if depths else 0
        scored_candidates.append((max_depth, potential_fact))

    if scored_candidates:
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        _, best_fact = scored_candidates[0]
        item["facts"].append(best_fact)


def _add_balancing_item(item, removed_item_data):
    item_type, removed_data = removed_item_data

    if item_type == 'rule':
        _add_balancing_rule(item, removed_data)
    elif item_type == 'fact':
        _add_balancing_fact(item, removed_data)


def do_remove_rule_or_fact(new_item, balance=False, max_removals=1):
    """
    Tries to make a problem unprovable by greedily removing up to
    max_removals rules or facts, prioritizing higher-depth results.
    """
    for _ in range(max_removals):
        if is_logically_consistent(set(new_item["facts"]), new_item["rules"], new_item["query"], 0):
            new_item["label"] = 0
            return new_item

        unprovable_candidates = []
        provable_candidates = []

        for i in range(len(new_item["rules"])):
            temp_rules = new_item["rules"][:i] + new_item["rules"][i + 1:]
            temp_facts = set(new_item["facts"])
            depths = calculate_depths(temp_facts, temp_rules)
            current_depth = max(depths.values()) if depths else 0
            if is_logically_consistent(temp_facts, temp_rules, new_item["query"], 0):
                unprovable_candidates.append((current_depth, 'rule', i))
            else:
                provable_candidates.append((current_depth, 'rule', i))

        facts_list = new_item["facts"]
        for i in range(len(facts_list)):
            temp_facts = set(facts_list[:i] + facts_list[i + 1:])
            temp_rules = new_item["rules"]
            depths = calculate_depths(temp_facts, temp_rules)
            current_depth = max(depths.values()) if depths else 0
            if is_logically_consistent(temp_facts, temp_rules, new_item["query"], 0):
                unprovable_candidates.append((current_depth, 'fact', i))
            else:
                provable_candidates.append((current_depth, 'fact', i))

        def _remove(candidates):
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_depth, item_type, item_index = candidates[0]
            if item_type == 'rule':
                removed_data = new_item["rules"].pop(item_index)
                removed_item = ('rule', removed_data)
            elif item_type == 'fact':
                removed_data = new_item["facts"].pop(item_index)
                removed_item = ('fact', removed_data)
            if balance:
                _add_balancing_item(new_item, removed_item)

        if unprovable_candidates:
            _remove(unprovable_candidates)
        elif provable_candidates:
            _remove(provable_candidates)
        else:
            break

    return None


def do_remove_rule(new_item, balance=False):
    random.shuffle(new_item["rules"])
    for rule_index in range(len(new_item["rules"])):
        temp_rules = copy.copy(new_item["rules"])
        temp_rules.pop(rule_index)
        if is_logically_consistent(new_item["facts"], temp_rules, new_item["query"], new_item["label"]):
            new_item["rules"] = temp_rules

            if balance:
                # is called after label is swapped to 0. so if we remove 'random' rule then that rule most likely query inside of it.
                # To counteract undeniably true trees, generate a tree which seemingly is undeniably true but really isn't.
                # Statistically, smaller head size is provable more likely, so generate counter-samples using add_redundant_rule as opposed to add_redundant_rule_gen to counteract that.
                add_redundant_rule(new_item, tail=new_item["query"], c=1)

            return new_item
    return None


def remove_rule(new_item, balance=False, preserve_depth=False):
    new_item["label"] = 0
    if not new_item["rules"]:
        return None

    if not preserve_depth:
        if do_remove_rule(new_item, balance):
            return new_item, 'r'

        if not balance:
            return None

        if alter_query(new_item, 0):
            return new_item, 'q'
    else:
        if do_remove_rule_or_fact(new_item, balance, max_removals=100):
            return new_item, 'r'

    return None


def add_rule(new_item, balance=False, prune_factor=1, preserve_depth=False):
    new_item["label"] = 1
    head = []

    if preserve_depth:
        if new_item["depth"] == 0:  # rule with no premises is a fact
            if new_item["facts"] and balance:
                new_item["facts"].remove(random.choice(new_item["facts"]))
            new_item["facts"].append(new_item["query"])
            return new_item

        provable_depths = calculate_depths(set(new_item["facts"]), new_item["rules"])
        head_pool_at_target = [p for p, d in provable_depths.items() if d == new_item["depth"] - 1]

        if head_pool_at_target:
            head_pool_other = [p for p, d in provable_depths.items() if d <= new_item["depth"] - 1]
            head_num = random.randint(1, min(3, len(head_pool_other)))
            guaranteed_head = random.sample(head_pool_at_target, 1)[0]
            head.append(guaranteed_head)
            if head_num > 1:
                remaining_candidates = [p for p in head_pool_other if p != guaranteed_head]
                if remaining_candidates:
                    head.extend(random.sample(remaining_candidates, min(len(remaining_candidates), head_num - 1)))

    if not head:
        potential_head = is_reachable_in_budget(new_item["facts"], new_item["rules"], new_item["preds"], max(0, new_item["depth"] - 1))
        head = random.sample(list(potential_head), min(len(potential_head), random.randint(1, 3)))

    new_rule = [head, new_item["query"]]
    new_item["rules"].append(new_rule)
    random.shuffle(new_item["rules"])

    if balance:
        prune_rules(new_item, c=prune_factor, tail_only=True)

    if is_logically_consistent(new_item["facts"], new_item["rules"], new_item["query"], new_item["label"]):
        return new_item
    return None


def prune_rules(new_item, c=1, needs_be_consistent=True, tail_only=False):
    random.shuffle(new_item["rules"])

    for i in range(len(new_item["rules"]) - 1, -1, -1):
        if (new_item["query"] in new_item["rules"][i][0] and not tail_only) or new_item["rules"][i][1] == new_item["query"]:
            tmp_rules = new_item["rules"][:i] + new_item["rules"][i + 1:]

            if not needs_be_consistent or is_logically_consistent(new_item["facts"], tmp_rules, new_item["query"], new_item["label"]):
                new_item["rules"] = tmp_rules
                c -= 1
                if c == 0:
                    break
    return new_item


def add_redundant_rule(new_item, tail=None, c=1):
    # Adds rules based on transitive closure
    random.shuffle(new_item["rules"])
    for head1, tail1 in new_item["rules"]:
        for head2, tail2 in new_item["rules"]:
            if len(head2) == 1 and tail1 == head2[0] and (tail is None or tail == tail2):
                new_rule = [head1, tail2]
                if new_rule not in new_item["rules"]:
                    new_item["rules"].append(new_rule)
                    c -= 1
                    if c == 0:
                        return new_item
    return None


def add_redundant_rule_gen(new_item, tail=None, not_tail=None, c=1):
    """
    Adds rules based on generalized transitive closure.
    If (H1 -> T1) and (H2 -> T2) exist, and T1 is in H2,
    adds rule ( (H2 without T1) + H1 -> T2 ).
    """
    rules_to_check = copy.copy(new_item["rules"])
    random.shuffle(rules_to_check)

    for head1, tail1 in rules_to_check:
        h1_set = set(head1)
        for head2, tail2 in rules_to_check:
            h2_set = set(head2)

            if tail1 in h2_set:
                new_head_set = (h2_set - {tail1}) | h1_set
                if tail2 in new_head_set:
                    continue
                if not new_head_set:
                    continue

                # Ensures [(A, B), T] and [(B, A), T] are treated as duplicates
                new_head_list = sorted(list(new_head_set))
                new_rule = [new_head_list, tail2]
                if (tail is None or tail == tail2) and (not_tail is None or not_tail != tail2):
                    rule_exists = False
                    for existing_rule in new_item["rules"]:
                        if existing_rule[1] == new_rule[1] and sorted(list(existing_rule[0])) == new_rule[0]:
                            rule_exists = True
                            break

                    if not rule_exists:
                        random.shuffle(new_rule[0])
                        # print(f"Adding redundant rule: {new_rule} from ({head1} -> {tail1}) and ({head2} -> {tail2})")  # Debug print
                        new_item["rules"].append(new_rule)
                        c -= 1
                        if c == 0:
                            return new_item
    return new_item


def counterfactual_heuristics(ds, heuristics=["r2"], balance=False, preserve_depth=False):
    print(f"computing heuristics: {heuristics}")
    extension = []
    statistics = lambda: {
        "q 0 -> 1": 0,
        "q 1 -> 0": 0,
        "qrr 1 -> 0": 0,
        "f 0 -> 1": 0,
        "f 1 -> 0": 0,
        "r 0 -> 1": 0,
        "r 1 -> 0": 0,
        "r 1 -> 1": 0,
        "rr 1 -> 1": 0,
        "rr 0 -> 0": 0,
    }

    total_stats = statistics()
    futures = []
    with ProcessPoolExecutor() as executor:
        for item in ds:
            futures.append(executor.submit(compute_heuristics, heuristics, item, statistics(), balance, preserve_depth))

        for future in futures:
            items, stats = future.result()
            extension.extend(items)
            for k, v in stats.items():
                total_stats[k] += v

    print(f"heuristic added labels: {total_stats}")
    return extension


def compute_heuristics(heuristics, item, statistics, balance=False, preserve_depth=False):
    inversed = []
    if item["label"] == 1:
        if "r2" in heuristics:
            if new_item := remove_rule(copy.deepcopy(item), balance=balance, preserve_depth=preserve_depth):
                statistics[new_item[1] + " 1 -> 0"] += 1
                inversed.append(new_item[0])

    elif item["label"] == 0:
        if "r2" in heuristics:
            # prune_factor = 1  =>  remove what we added
            if new_item := add_rule(copy.deepcopy(item), balance=balance, prune_factor=1, preserve_depth=preserve_depth):
                statistics["r 0 -> 1"] += 1
                inversed.append(new_item)

    return inversed, statistics


def run_tests():
    """Runs all test cases and prints the results."""
    print("Running tests...")

    facts1 = []
    rules1 = [(['B'], 'C'), ([], 'B'), (['B'], 'D'), (['C', 'D'], 'E')]
    query1 = 'E'
    assert solve(facts1, rules1, query1, break_early=True, include_copy=True, eager=True)[0] == ['B', 'C', 'D', 'E']
    assert solve(facts1, rules1, query1, break_early=True, include_copy=False, eager=True)[0] == ['B', 'C', 'D', 'E']
    assert solve(facts1, rules1, query1, break_early=False, include_copy=True, eager=True)[0] == ['B', 'C', 'D', 'E']
    assert solve(facts1, rules1, query1, break_early=False, include_copy=False, eager=True)[0] == ['B', 'C', 'D', 'E']

    random.seed(1)
    assert solve(facts1, rules1, query1, break_early=True, include_copy=True, eager=False)[0] == ['B', 'C', 'D', 'E']
    random.seed(4)
    assert solve(facts1, rules1, query1, break_early=True, include_copy=True, eager=False)[0] == ['B', 'D', 'C', 'E']

    facts2 = ['A']
    rules2 = [(['A'], 'B'), ([], 'Z'), (['B'], 'C'), (['A'], 'D'), (['B'], 'A')]
    query2 = 'B'
    assert solve(facts2, rules2, query2, break_early=True, include_copy=True, eager=True)[0] == ['A', 'B']
    assert solve(facts2, rules2, query2, break_early=True, include_copy=False, eager=True)[0] == ['B']
    assert solve(facts2, rules2, query2, break_early=False, include_copy=True, eager=True)[0] == ['A', 'B', 'Z', 'C', 'D']
    assert solve(facts2, rules2, query2, break_early=False, include_copy=False, eager=True)[0] == ['B', 'Z', 'C', 'D']

    facts3 = ['A']
    rules3 = [(['A'], 'B'), (['A'], 'C'), (['B'], 'D'), (['D'], 'E')]
    query3 = 'D'
    labels = ['A', 'B', 'C', 'D', 'E']
    smooth_labels = [['A'], ['B', 'C'], ['C', 'D'], ['D'], ['E']]
    assert solve(facts3, rules3, query3, break_early=False, include_copy=True, eager=True) == (labels, smooth_labels)
    assert solve(facts3, rules3, query3, break_early=False, include_copy=False, eager=True) == (labels[1:], smooth_labels[1:])
    assert solve(facts3, rules3, query3, break_early=True, include_copy=True, eager=True) == (labels[:-1], smooth_labels[:-1])
    assert solve(facts3, rules3, query3, break_early=True, include_copy=False, eager=True) == (labels[1:-1], smooth_labels[1:-1])

    facts4 = ['A', 'B']
    rules4 = [(['C'], 'D')]
    query4 = 'D'
    assert solve(facts4, rules4, query4, include_copy=True)[1] == [['A'], ['B']]
    assert solve(facts4, rules4, query4, include_copy=False)[1] == []
    print("All tests passed!")


if __name__ == "__main__":
    run_tests()
