# make sure files exist because scripts only append to files

# to evaluate hyperparameters

#python sample.py --vocab_file vocab.txt --output_file /media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_lp.jsonl --min_pred_num 5 --max_pred_num 30 --algo LP --balance_by_depth --example_num 500 --max_depth 6
#python sample.py --vocab_file vocab.txt --output_file /media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_lp_balanced.jsonl --min_pred_num 5 --max_pred_num 30 --algo LP_BALANCED --balance_by_depth --example_num 500 --max_depth 6
#python sample.py --vocab_file vocab.txt --output_file /media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp.jsonl --min_pred_num 5 --max_pred_num 30 --algo RP --balance_by_depth --example_num 500 --max_depth 6
#python sample.py --vocab_file vocab.txt --output_file /media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp_balanced.jsonl --min_pred_num 5 --max_pred_num 30 --algo RP_BALANCED --balance_by_depth --example_num 500 --max_depth 6

# to evaluate best models

#python sample.py --vocab_file vocab.txt --output_file /media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_lp_balanced_deep.jsonl --min_pred_num 5 --max_pred_num 30 --algo LP_BALANCED --balance_by_depth --example_num 500 --max_depth 12
#python sample.py --vocab_file vocab.txt --output_file /media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_lp_balanced_deep_60.jsonl --min_pred_num 5 --max_pred_num 60 --algo LP_BALANCED --balance_by_depth --example_num 500 --max_depth 12
#python sample.py --vocab_file vocab.txt --output_file /media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp_balanced_deep.jsonl --min_pred_num 5 --max_pred_num 30 --algo RP_BALANCED --balance_by_depth --example_num 500 --max_depth 12
#python sample.py --vocab_file vocab.txt --output_file /media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp_balanced_deep_60.jsonl --min_pred_num 5 --max_pred_num 60 --algo RP_BALANCED --balance_by_depth --example_num 500 --max_depth 12
#python sample.py --vocab_file vocab.txt --output_file /media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_lp_star_balanced_deep.jsonl --min_pred_num 5 --max_pred_num 30 --algo LP_STAR_BALANCED --balance_by_depth --example_num 500 --max_depth 12
#python sample.py --vocab_file vocab.txt --output_file /media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_lp_star_balanced_deep_60.jsonl --min_pred_num 5 --max_pred_num 60 --algo LP_STAR_BALANCED --balance_by_depth --example_num 500 --max_depth 12

# need to update sample_one_rule function beforehand
#python sample.py --vocab_file vocab.txt --output_file /media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp_balanced_3_premise.jsonl --min_pred_num 5 --max_pred_num 30 --algo RP_BALANCED --balance_by_depth --example_num 500 --max_depth 6
#python sample.py --vocab_file vocab.txt --output_file /media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp_balanced_2_premise.jsonl --min_pred_num 5 --max_pred_num 30 --algo RP_BALANCED --balance_by_depth --example_num 500 --max_depth 6
#python sample.py --vocab_file vocab.txt --output_file /media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp_balanced_1_premise.jsonl --min_pred_num 5 --max_pred_num 30 --algo RP_BALANCED --balance_by_depth --example_num 500 --max_depth 6
#python sample.py --vocab_file vocab.txt --output_file /media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp_balanced_1+2_premise.jsonl --min_pred_num 5 --max_pred_num 30 --algo RP_BALANCED --balance_by_depth --example_num 500 --max_depth 6
#python sample.py --vocab_file vocab.txt --output_file /media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp_balanced_2+3_premise.jsonl --min_pred_num 5 --max_pred_num 30 --algo RP_BALANCED --balance_by_depth --example_num 500 --max_depth 6
