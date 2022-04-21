"""
Computes confidence intervals and p-values (for comparison of two desired scores)
for ROC and PR AUCs of multiple scores.
"""


from typing import List
import pandas as pd
import numpy as np
from cached_property import cached_property
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, average_precision_score


class BootstrapROCPR:
    def __init__(self,
                 y_true: pd.Series,
                 scores: List[pd.Series],
                 n_draws: int = 10000,
                 alpha: int = 5,
                 n_jobs: int = 20,
                 compare: tuple = (0, 1)):

        self.y_true = y_true
        self.scores = scores
        self.n_draws = n_draws
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.compare = compare

    @cached_property
    def roc_auc_all_obs(self):
        return {score.name: roc_auc_score(self.y_true, score) for score in self.scores}

    @cached_property
    def pr_auc_all_obs(self):
        return {score.name: average_precision_score(self.y_true, score) for score in self.scores}

    def bootstrap_auc(self, random_state):

        n = self.y_true.size

        roc_aucs = {score.name:
                    roc_auc_score(self.y_true.sample(n=n, replace=True, random_state=random_state),
                                  score.sample(n=n, replace=True, random_state=random_state))
                    for score in self.scores}

        pr_aucs = {score.name:
                   average_precision_score(self.y_true.sample(n=n, replace=True, random_state=random_state),
                                           score.sample(n=n, replace=True, random_state=random_state))
                   for score in self.scores}

        return {'roc': roc_aucs, 'pr': pr_aucs}

    @cached_property
    def auc_scores(self):
        aucs = Parallel(n_jobs=self.n_jobs)(delayed(self.bootstrap_auc)(r) for r in range(self.n_draws))

        roc_aucs = {score.name: [a['roc'][score.name] for a in aucs] for score in self.scores}
        pr_aucs = {score.name: [a['pr'][score.name] for a in aucs] for score in self.scores}

        return roc_aucs, pr_aucs

    @cached_property
    def roc_auc_confidence_interval(self):
        roc_aucs, pr_aucs = self.auc_scores

        roc_ci = {score: (np.percentile(roc_aucs[score], 0.5 * self.alpha),
                          np.percentile(roc_aucs[score], 100 - 0.5 * self.alpha))
                  for score in roc_aucs.keys()}

        return roc_ci

    @cached_property
    def pr_auc_confidence_interval(self):
        roc_aucs, pr_aucs = self.auc_scores

        pr_ci = {score: (np.percentile(pr_aucs[score], 0.5 * self.alpha),
                         np.percentile(pr_aucs[score], 100 - 0.5 * self.alpha))
                 for score in pr_aucs.keys()}

        return pr_ci

    @cached_property
    def roc_auc_diffs(self):
        roc_aucs, pr_aucs = self.auc_scores

        # roc diffs
        roc_aucs_1 = roc_aucs[self.scores[self.compare[0]].name]
        roc_aucs_2 = roc_aucs[self.scores[self.compare[1]].name]
        roc_auc_diffs = list(np.array(roc_aucs_1) - np.array(roc_aucs_2))

        return roc_auc_diffs

    @cached_property
    def pr_auc_diffs(self):
        roc_aucs, pr_aucs = self.auc_scores

        # pr auc diffs
        pr_aucs_1 = pr_aucs[self.scores[self.compare[0]].name]
        pr_aucs_2 = pr_aucs[self.scores[self.compare[1]].name]
        pr_auc_diffs = list(np.array(pr_aucs_1) - np.array(pr_aucs_2))

        return pr_auc_diffs

    @cached_property
    def roc_diff_pval(self):
        # ToDo: Could be extended to two-sided tests or tests "the other way round"
        roc_auc_1 = self.roc_auc_all_obs[self.scores[self.compare[0]].name]
        roc_auc_2 = self.roc_auc_all_obs[self.scores[self.compare[1]].name]
        std_diff = np.std(self.roc_auc_diffs)
        t = (roc_auc_1 - roc_auc_2) / std_diff
        return 1 - norm.cdf(t)

    @cached_property
    def pr_diff_pval(self):
        pr_auc_1 = self.pr_auc_all_obs[self.scores[self.compare[0]].name]
        pr_auc_2 = self.pr_auc_all_obs[self.scores[self.compare[1]].name]
        std_diff = np.std(self.pr_auc_diffs)
        t = (pr_auc_1 - pr_auc_2) / std_diff
        return 1 - norm.cdf(t)
