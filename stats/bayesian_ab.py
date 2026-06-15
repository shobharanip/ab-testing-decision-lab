"""
Bayesian A/B Testing Module
CEO-Track Portfolio | A/B Testing Decision Lab
"""
import numpy as np
import scipy.stats as stats
from dataclasses import dataclass
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class BayesianResult:
      """Result of a Bayesian A/B test."""
      prob_b_beats_a: float
      expected_lift_b: float
      ci_lower: float
      ci_upper: float
      expected_loss_b: float
      recommendation: str
      confidence_level: str
      n_a: int
      n_b: int
      rate_a: float
      rate_b: float


class BayesianABTest:
      """
          Bayesian A/B test using Beta-Binomial conjugate model.
              Answers: What is the probability treatment beats control?
                  """

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0,
                                  n_samples: int = 50000):
                                            self.alpha_prior = alpha_prior
                                            self.beta_prior = beta_prior
                                            self.n_samples = n_samples

    def analyze(self, n_a: int, conv_a: int, n_b: int, conv_b: int,
                                mde: float = 0.01) -> BayesianResult:
                                          """Run Bayesian A/B test analysis with decision recommendation."""
                                          post_a = stats.beta(self.alpha_prior + conv_a, self.beta_prior + (n_a - conv_a))
                                          post_b = stats.beta(self.alpha_prior + conv_b, self.beta_prior + (n_b - conv_b))
                                          samples_a = post_a.rvs(self.n_samples)
                                          samples_b = post_b.rvs(self.n_samples)
                                          prob_b_beats_a = float(np.mean(samples_b > samples_a))
                                          lift_samples = (samples_b - samples_a) / np.maximum(samples_a, 1e-8)
                                          expected_lift = float(np.mean(lift_samples))
                                          ci_lower = float(np.percentile(lift_samples, 2.5))
                                          ci_upper = float(np.percentile(lift_samples, 97.5))
                                          loss_b = float(np.mean(np.maximum(samples_a - samples_b, 0)))
                                          rec, conf = self._recommend(prob_b_beats_a, expected_lift, mde)
                                          return BayesianResult(
                                              prob_b_beats_a=prob_b_beats_a,
                                              expected_lift_b=expected_lift,
                                              ci_lower=ci_lower,
                                              ci_upper=ci_upper,
                                              expected_loss_b=loss_b,
                                              recommendation=rec,
                                              confidence_level=conf,
                                              n_a=n_a, n_b=n_b,
                                              rate_a=conv_a/n_a, rate_b=conv_b/n_b
                                          )

    def _recommend(self, prob: float, lift: float, mde: float) -> Tuple[str, str]:
              """Generate shipping decision based on posterior probability."""
              if prob >= 0.99 and abs(lift) >= mde:
                            return "SHIP TREATMENT", "VERY HIGH (99%+)"
elif prob >= 0.95 and abs(lift) >= mde:
            return "SHIP TREATMENT", "HIGH (95%+)"
elif prob >= 0.90:
            return "EXTEND TEST", "MODERATE (90%+)"
elif prob <= 0.10:
            return "KEEP CONTROL", "HIGH (90%+ control wins)"
else:
            return "CONTINUE TESTING", "INCONCLUSIVE"


class RevenueImpactCalculator:
      """Translates A/B test lift into ARR/MRR business impact."""

    def __init__(self, monthly_visitors: int, avg_order_value: float,
                                  current_conversion_rate: float):
                                            self.monthly_visitors = monthly_visitors
                                            self.aov = avg_order_value
                                            self.baseline_rate = current_conversion_rate

    def calculate(self, lift_fraction: float) -> dict:
              """Calculate monthly and annual revenue impact."""
              baseline_rev = self.monthly_visitors * self.baseline_rate * self.aov
              new_rate = self.baseline_rate * (1 + lift_fraction)
              new_rev = self.monthly_visitors * new_rate * self.aov
              monthly_lift = new_rev - baseline_rev
              return {
                  'baseline_monthly_revenue': round(baseline_rev, 2),
                  'new_monthly_revenue': round(new_rev, 2),
                  'monthly_lift': round(monthly_lift, 2),
                  'annual_arr_impact': round(monthly_lift * 12, 2),
                  'lift_pct': round(lift_fraction * 100, 2)
              }


if __name__ == '__main__':
      test = BayesianABTest()
      result = test.analyze(n_a=12483, conv_a=477, n_b=12483, conv_b=515)
      print(f"P(B>A): {result.prob_b_beats_a:.1%}")
      print(f"Lift: {result.expected_lift_b:.1%} CI[{result.ci_lower:.1%}, {result.ci_upper:.1%}]")
      print(f"Decision: {result.recommendation} ({result.confidence_level})")
      calc = RevenueImpactCalculator(50000, 150.0, 0.0382)
      impact = calc.calculate(result.expected_lift_b)
      print(f"ARR Impact: ${impact['annual_arr_impact']:,.0f}")
