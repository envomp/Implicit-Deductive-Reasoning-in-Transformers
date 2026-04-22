# Grid search results for selected hyperparameters

Models were trained on LP at first,
before we knew much about the shortcuts.

---

Curriculum learning introduces complex training data sequentially.
We compared random sampling (random curriculum) against deterministic sampling (eager curriculum).

| Run ID                      | LP (%) | LP* (%) | RP (%) |
|:----------------------------|-------:|--------:|-------:|
| direct eager curriculum     |   70.1 |    65.9 |   65.2 |
| direct random curriculum    |   69.8 |    65.8 |   63.2 |
| direct                      |   81.7 |    55.1 |   63.0 |
| r2 direct eager curriculum  |   69.4 |    65.6 |   70.6 |
| r2 direct random curriculum |   70.9 |    68.1 |   70.2 |
| r2 direct                   |   85.6 |    67.6 |   77.6 |

When utilizing the r2 heuristic, the standard “no curriculum” approach outperformed curriculum-based methods.

---

We distinguish between random-cot, which samples the next step stochastically from all valid deductions, and
eager-cot, which deterministically selects the deduction from the first applicable rule in a fixed left-to-right traversal.

| Run ID                       | LP (%) | LP* (%) | RP (%) |
|:-----------------------------|-------:|--------:|-------:|
| random-cot eager curriculum  |   95.8 |    90.0 |   56.9 |
| random-cot random curriculum |   94.6 |    87.3 |   55.2 |
| random-cot                   |   98.1 |    90.8 |   65.5 |
| eager-cot eager curriculum   |   97.7 |    92.2 |   76.2 |
| eager-cot random curriculum  |   93.0 |    83.8 |   70.7 |
| eager-cot                    |   99.0 |    94.0 |   68.2 |

| Run ID                          | LP (%) | LP* (%) | RP (%) |
|:--------------------------------|-------:|--------:|-------:|
| r2 random-cot eager curriculum  |   97.9 |    93.5 |   87.2 |
| r2 random-cot random curriculum |   97.4 |    93.1 |   85.1 |
| r2 random-cot                   |   96.8 |    91.6 |   78.2 |
| r2 eager-cot eager curriculum   |   96.2 |    92.2 |   77.6 |
| r2 eager-cot random curriculum  |   96.8 |    92.4 |   83.4 |
| r2 eager-cot                    |   97.7 |    95.0 |   87.8 |

Eager selection combined with the r2 heuristic and no curriculum yielded the highest accuracy.

---

Out of "fancy" things, only corrective significantly outperformed the baseline.
Usually, bidirectional models are first trained under masked objective, but we found it provides no further benefit.
Also, when we have custom shuffling where adversarial pairs are in same batch, then that didn't improve performance either.

| Run ID               | LP (%) | LP* (%) | RP (%) |
|:---------------------|-------:|--------:|-------:|
| r2 direct corrective |   95.5 |    81.9 |   78.9 |
| r2 direct mixed      |   89.9 |    75.1 |   67.1 |
| r2 direct shuffle    |   81.8 |    71.8 |   75.4 |
| r2 direct masked     |   73.4 |    59.0 |   80.7 |
| r2 direct            |   85.4 |    66.0 |   73.9 |

| Run ID            | LP (%) | LP* (%) | RP (%) |
|:------------------|-------:|--------:|-------:|
| r2 cot grpo       |   96.5 |    92.4 |   86.9 |
| r2 cot corrective |   99.0 |    95.9 |   89.9 |
| r2 cot mixed      |   96.9 |    91.5 |   72.2 |
| r2 cot shuffle    |   98.7 |    95.3 |   90.1 |
| r2 cot masked     |   98.2 |    94.8 |   88.4 |
| r2 cot            |   97.5 |    94.7 |   88.1 |

---

Reducing precision below float32 prevented learning, as did disabling weight decay on RMSNorm weights.

| Run ID              | LP (%) | LP* (%) | RP (%) |
|:--------------------|-------:|--------:|-------:|
| r2 cot -wd_on_lnorm |   51.6 |    49.5 |   49.9 |
| r2 cot bfloat16     |   50.0 |    50.0 |   50.0 |

---

Some grid search results are here when trained with corrective objective, 
other preliminary findings are visualized in plot.ipynb file.
Taken together when also considering the hyperparameters used when training large LLMS,
we determined the best hyperparameters which are in the paper.

| Run ID         | LP (%) | LP* (%) | RP (%) |
|:---------------|-------:|--------:|-------:|
| direct wd=0.4  |   89.6 |    74.5 |   70.8 |
| direct wd=0.2  |   92.1 |    75.2 |   81.3 |
| direct wd=0.1  |   95.5 |    81.9 |   78.9 |
| direct wd=0.05 |   93.3 |    78.3 |   85.8 |
| cot wd=0.4     |   96.9 |    92.9 |   76.0 |
| cot wd=0.2     |   97.4 |    92.6 |   83.6 |
| cot wd=0.1     |   99.0 |    95.9 |   89.9 |
| cot wd=0.05    |   98.7 |    94.8 |   89.7 |

| Run ID             | LP (%) | LP* (%) | RP (%) |
|:-------------------|-------:|--------:|-------:|
| direct beta2=0.999 |   94.2 |    79.7 |   68.9 |
| direct beta2=0.995 |   95.1 |    77.7 |   82.0 |
| direct beta2=0.99  |   95.5 |    81.9 |   78.9 |
| direct beta2=0.98  |   94.1 |    78.0 |   86.0 |
| cot beta2=0.999    |   98.4 |    94.5 |   77.8 |
| cot beta2=0.995    |   98.6 |    93.5 |   91.5 |
| cot beta2=0.99     |   99.0 |    95.9 |   89.9 |
| cot beta2=0.98     |   98.3 |    94.8 |   88.5 |

| Run ID                | LP (%) | LP* (%) | RP (%) |
|:----------------------|-------:|--------:|-------:|
| direct lr=0.01..1e-6  |   94.4 |    78.6 |   67.8 |
| direct lr=0.01..1e-5  |   86.3 |    69.3 |   74.0 |
| direct lr=0.01..5e-5  |   92.9 |    75.1 |   82.0 |
| direct lr=0.01..1e-4  |   95.5 |    81.9 |   78.9 |
| direct lr=0.01..5e-4  |   90.6 |    72.5 |   83.1 |
| direct lr=0.01..0.001 |   93.7 |    79.9 |   66.7 |
| cot lr=0.01..1e-6     |   98.9 |    95.8 |   82.4 |
| cot lr=0.01..1e-5     |   97.1 |    92.0 |   84.8 |
| cot lr=0.01..5e-5     |   97.9 |    93.2 |   89.4 |
| cot lr=0.01..1e-4     |   99.0 |    95.9 |   89.9 |
| cot lr=0.01..5e-4     |   97.8 |    94.0 |   86.7 |
| cot lr=0.01..0.001    |   97.3 |    92.7 |   83.4 |

| Run ID                 | LP (%) | LP* (%) | RP (%) |
|:-----------------------|-------:|--------:|-------:|
| direct batch_size=1000 |   95.8 |    78.0 |   82.7 |
| direct batch_size=500  |   95.5 |    81.9 |   78.9 |
| direct batch_size=250  |   89.8 |    71.9 |   82.8 |
| cot batch_size=1000    |   98.9 |    96.7 |   90.9 |
| cot batch_size=500     |   99.0 |    95.9 |   89.9 |
| cot batch_size=250     |   97.9 |    92.5 |   91.0 |


