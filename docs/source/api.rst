.. _api_ref:

.. currentmodule:: bayesplain

API reference
=============

The analyses
------------

Eleven functions, each returning a :class:`Result`. The frequentist
counterpart each one is paired with is named in its own documentation.

.. autosummary::
   :toctree: generated/

   proportion
   compare_proportions
   rate
   compare_rates
   contingency
   categories
   mean
   compare_means
   correlation
   regression
   compare_groups


The result object
-----------------

Every analysis returns one of these, with the same methods throughout, so the
interface learned for one carries over to all of them.

.. autosummary::
   :toctree: generated/

   Result
   Decision
   BayesFactor
   Prediction


Priors
------

.. currentmodule:: bayesplain.priors

Named presets, so nobody has to type a shape parameter to get started, and
nobody can hide which assumption they used.

.. autosummary::
   :toctree: generated/

   BetaPrior
   ConcentrationPrior
   EffectSizePrior
   CorrelationPrior
   RatePrior
   resolve_proportion
   resolve_table
   resolve_effect_size
   resolve_correlation
   resolve_rate
   from_previous_study
   from_previous_period
   describe
   available


Frequentist twins
-----------------

.. currentmodule:: bayesplain.frequentist

The conventional tests, computed and printed alongside every Bayesian result.
Usable on their own when you only want the classical answer.

.. autosummary::
   :toctree: generated/

   FrequentistTwin
   one_proportion
   two_proportions
   several_proportions
   one_rate
   two_rates
   chi_square_independence
   goodness_of_fit
   one_mean
   two_means
   paired_means
   correlation
   simple_regression
   one_way_anova


Teaching tools
--------------

.. currentmodule:: bayesplain.teach

Pedagogy-only helpers that show a mechanism the analysis functions deliberately
hide. Nothing here belongs in a research package; all of it belongs in a first
course.

.. autosummary::
   :toctree: generated/

   natural_frequencies
   grid_posterior
   binomial_likelihood
   sequential
   precision_planning
   precision_planning_mean
   NaturalFrequencies
   GridPosterior
   SequentialUpdate
   PrecisionPlan


Datasets
--------

.. currentmodule:: bayesplain.datasets

.. autosummary::
   :toctree: generated/

   available
   describe
   load
   load_collisions
   load_tracts
   load_blockgroups
   load_parcels


Configuration
-------------

.. currentmodule:: bayesplain

.. autosummary::
   :toctree: generated/

   get_seed
   set_seed
   get_draws
   set_draws


Core mathematics
================

Layer 0: pure functions, no objects, no printing, nothing that knows it is
being used to teach. Documented because the derivations belong somewhere
readable, and because a methods audience will want to check them.

Proportions
-----------

.. currentmodule:: bayesplain.core.beta_binomial

.. autosummary::
   :toctree: generated/

   posterior
   draws
   log_marginal_likelihood
   log_bayes_factor_point_null
   prior_predictive_mean
   validate_counts


Rates
-----

.. currentmodule:: bayesplain.core.gamma_poisson

.. autosummary::
   :toctree: generated/

   posterior
   ratio_posterior
   predictive
   log_marginal_likelihood
   log_bayes_factor_point_null
   log_bayes_factor_equal_rates
   validate_events


Contingency tables and categories
---------------------------------

.. currentmodule:: bayesplain.core.dirichlet_multinomial

.. autosummary::
   :toctree: generated/

   log_bayes_factor_independence
   posterior_cell_draws
   log_bayes_factor_goodness_of_fit
   cramers_v
   cohens_w
   log_odds_ratio
   log_multivariate_beta
   validate_table


Means
-----

.. currentmodule:: bayesplain.core.normal_t

.. autosummary::
   :toctree: generated/

   mean_posterior
   pooled_difference_posterior
   difference_draws
   log_bayes_factor_ttest
   summarise
   validate_sample


Correlation
-----------

.. currentmodule:: bayesplain.core.correlation

.. autosummary::
   :toctree: generated/

   log_sampling_density
   log_prior_density
   posterior_on_grid
   log_bayes_factor
   rho_grid
   validate_pair


Regression
----------

.. currentmodule:: bayesplain.core.regression

.. autosummary::
   :toctree: generated/

   fit_line
   LineFit
   slope_posterior
   mean_response_posterior
   predictive
   log_bayes_factor_slope


Partial pooling
---------------

.. currentmodule:: bayesplain.core.hierarchical

.. autosummary::
   :toctree: generated/

   shrink
   shrink_proportions
   between_group_variance


Summarising a posterior
-----------------------

.. currentmodule:: bayesplain.core.intervals

.. autosummary::
   :toctree: generated/

   interval
   hdi_from_draws
   eti_from_draws
   probability_from_draws
   monte_carlo_se


Grid approximation
------------------

.. currentmodule:: bayesplain.core.grid

.. autosummary::
   :toctree: generated/

   normalize_log_density
   sample_from_grid
   grid_mean
   grid_quantile
