#import "@preview/slydst:0.1.5": *

#show: slides.with(
  title: "Compound Poisson: Gamma, Tweedie Pure Premium, and Exposure Weights",
  subtitle: "Deriving the Tweedie pure-premium model from first principles",
  date: "August 11, 2026",
  authors: "Thomas Bury",
  layout: "medium",
  ratio: 4/3,
  title-color: none,
)

#show raw.where(block: true): set block(
  fill: silver.lighten(65%),
  width: 100%,
  inset: 1em,
)

#show raw.where(block: false): set text(fill: navy)

== Overview

This note derives the connections between:

+ a Poisson model for claim counts
+ a Gamma model for individual claim severities
+ the compound Poisson--Gamma distribution for aggregate claim cost
+ the Tweedie distribution with power $1 < p < 2$
+ the pure premium (loss cost) as a rate per unit exposure
+ the appearance of `sample_weight = exposure` when fitting a Tweedie pure-premium model
+ the exact equivalence, in the Poisson case, between claim counts with a log-exposure offset and observed claim frequency with exposure as sample weight.

#definition(title: "Exposure scaling")[
  Exposure scales the *amount* of risk observed. If claim arrivals form a
  Poisson process, both the expected aggregate claim amount and its variance
  scale linearly with exposure.
]

For $1 < p < 2$, the Tweedie pure-premium distribution follows exactly from a
compound Poisson model with Gamma severities.

= 1. Why notation matters

== Two common uses of the symbol $lambda$

A common source of confusion is the symbol $lambda$.

In many texts,

$ N ~ op("Poisson")(lambda) $

uses $lambda$ for the *Poisson mean*, i.e. the expected number of events
over the observation interval.

In insurance texts one also often sees

$ N_i ~ op("Poisson")(w_i lambda_i) $

where $lambda_i$ is now interpreted as a *claim-frequency rate per unit
exposure*.

Both conventions are valid, but using the same symbol for an absolute expected
count and for a rate easily obscures the role of exposure.

This note uses two different symbols.

== Core notation

=== 1.1 Notation

For policy or observation $i$:

#table(
  columns: 3,
  align: left,
  stroke: 0.5pt + gray,
  table.header(
    [Symbol], [Meaning], [Typical unit],
  ),
  [$X_i$], [rating variables / covariates], [--],
  [$e_i > 0$], [exposure], [policy-years],
  [$nu_i$], [claim-frequency (a *rate*)], [claims / policy-year],
  [$Lambda_i$], [expected claim *count* over observed exposure], [claims],
  [$N_i$], [observed number of claims], [claims],
  [Z sub {ij}], [amount of claim j], [EUR / claim],
  [$zeta_i$], [expected severity], [EUR / claim],
  [$S_i$], [observed aggregate claim amount], [EUR],
  [$mu_i$], [expected pure-premium *rate*], [EUR / policy-year],
  [$M_i$], [expected aggregate claim amount], [EUR],
  [$R_i$], [observed pure premium $S_i / e_i$], [EUR / policy-year],
  [$p$], [Tweedie power], [dimensionless],
  [$phi$], [base Tweedie dispersion parameter], [model-dependent units],
)

The rate and count satisfy

$ Lambda_i = e_i nu_i $

and pure premium = claim frequency x severity:

$ mu_i = nu_i zeta_i $

Consequently: $M_i = e_i mu_i$

= 2. Frequency: rate versus expected count

== Frequency: rate versus expected count

Assume claim arrivals follow a Poisson process conditional on the policy
characteristics $X_i$. Let

$ nu_i = nu(X_i) $

denote the claim-frequency rate. If exposure is measured in policy-years,
then $nu_i$ has units

$ "claims" / "policy-year" $

The expected number of claims during exposure $e_i$ is

$ Lambda_i = e_i nu_i $

We model

$ N_i | X_i, e_i ~ op("Poisson")(Lambda_i) = op("Poisson")(e_i nu_i) $

Thus,

$ bb("E")[N_i | X_i, e_i] = e_i nu_i $

and because a Poisson random variable has variance equal to its mean,

$ op("Var")(N_i | X_i, e_i) = e_i nu_i $

== Concrete numerical illustration

Suppose a fixed frequency $nu_i = 0.10$ "claims per policy-year".

For a full year,
$ e_i = 1 => Lambda_i = 0.10 $

For half a year,
$ e_i = 0.5 => Lambda_i = 0.05 $

For three months,
$ e_i = 0.25 => Lambda_i = 0.025 $

The annualized rate stays the same. Exposure converts it into the expected
count over the observed risk period. Unit exposure means $e_i = 1$.

= 3. Severity model

== Gamma severity model

Let $Z_(i 1), Z_(i 2), ...$ be individual claim amounts for policy $i$.
Conditional on $X_i$, assume the claim amounts are iid and independent of
$N_i$. Write

$ zeta_i = bb("E")[Z_("ij") | X_i] $

for mean severity.

For the exact Tweedie compound-Poisson representation we assume Gamma
severities, which have strictly positive support.

To avoid ambiguity between Gamma *scale* and *rate* conventions, this note
uses a shape-rate parameterization:

$ Z_("ij") | X_i ~ op("Gamma")(gamma, c_i) $

where $gamma > 0$ is the shape and $c_i > 0$ is the rate. Therefore,

$ bb("E")[Z_("ij") | X_i] = frac(gamma, c_i) = zeta_i $
$ op("Var")(Z_("ij") | X_i) = frac(gamma, c_i^2) $
$ bb("E")[Z_("ij")^2 | X_i] = frac(gamma (gamma + 1), c_i^2) $

= 4. Aggregate claim cost

== Compound Poisson aggregate loss

Define aggregate claim cost

$ S_i = sum_(j = 1)^(N_i) Z_("ij") $

with the usual convention that $S_i = 0$ when $N_i = 0$. We have

$ N_i | X_i, e_i ~ op("Poisson")(e_i nu_i) $

and iid Gamma severities. Therefore $S_i$ has a compound Poisson--Gamma
distribution. This is the classical collective risk model underlying the
Tweedie distribution for $1 < p < 2$; see Jørgensen and de Souza [2] and
Delong, Lindholm and Wüthrich [1].

= 5. Mean aggregate claim cost

== Mean aggregate claim cost

Condition first on the number of claims. Given $N_i$,

$ bb("E")[S_i | N_i, X_i, e_i] = N_i zeta_i $

Taking expectation again,

$ bb("E")[S_i | X_i, e_i] = bb("E")[N_i | X_i, e_i] zeta_i = e_i nu_i zeta_i $

Define the expected pure-premium rate

$ mu_i = nu_i zeta_i $

Hence

$ bb("E")[S_i | X_i, e_i] = e_i mu_i $

This is the actuarial identity

$ "pure premium rate" = "claim-frequency rate" times "mean severity" $

= 6. Variance of aggregate claim cost

== Law of total variance

Using the law of total variance,

$ op("Var")(S_i | X_i, e_i) = bb("E")[op("Var")(S_i | N_i, X_i, e_i) | X_i, e_i] + op("Var")(bb("E")[S_i | N_i, X_i, e_i] | X_i, e_i) $

Given $N_i$,

$ op("Var")(S_i | N_i, X_i, e_i) = N_i op("Var")(Z_("ij") | X_i) $

so the first term is $bb("E")[N_i | X_i, e_i] op("Var")(Z_("ij") | X_i)$. The
second term is $zeta_i^2 op("Var")(N_i | X_i, e_i)$.

Because $N_i$ is Poisson,

$ bb("E")[N_i | X_i, e_i] = op("Var")(N_i | X_i, e_i) = e_i nu_i $

Therefore

$ op("Var")(S_i | X_i, e_i) = e_i nu_i [op("Var")(Z_("ij") | X_i) + zeta_i^2] $

== General compound Poisson variance identity

But

$ op("Var")(Z_("ij")) + (bb("E")[Z_("ij")])^2 = bb("E")[Z_("ij")^2] $

Hence

$ op("Var")(S_i | X_i, e_i) = e_i nu_i bb("E")[Z_("ij")^2 | X_i] $

This variance identity holds for any compound Poisson model.

#definition(title: "Linear exposure scaling")[
  The exposure dependence is already explicit:
  $op("Var")(S_i | X_i, e_i) prop e_i$.
  This linear scaling follows from the Poisson-process assumption.
]

= 7. The exact Tweedie connection

== Distributional equivalence

For $1 < p < 2$, the Tweedie distribution is exactly a compound Poisson
distribution with Gamma jumps.

Delong, Lindholm and Wüthrich [1, Sections 2.1--2.2, Proposition 2.1] give
a derivation by comparing moment-generating functions.
They start from a compound Poisson--Gamma model with claim count mean equal
to exposure times claim frequency and identify the corresponding Tweedie
parameters.

Jørgensen and de Souza [2] is a foundational actuarial reference for the same
compound-Poisson Tweedie process.

= 8. Mean-dispersion notation for the Tweedie distribution

== Tweedie parameterization

Define

$ Y ~ op("Tw")_p (m, d) $

to mean a Tweedie random variable satisfying

$ bb("E")[Y] = m "  and  " op("Var")(Y) = d m^p $

Here $m$ is the mean, $d$ is the actual dispersion attached to this
observation, and $p in (1, 2)$ is the Tweedie power.

Later, for a pure-premium observation with exposure $e_i$, we will obtain

$ d_i = frac(phi, e_i) $

This notation separates the *observation-specific dispersion*
$d_i$ from the common base parameter $phi$.

= 9. Mapping Tweedie parameters to the compound Poisson--Gamma model

== Parameter correspondence

For $Y ~ op("Tw")_p (mu, phi)$ with $1 < p < 2$, the associated compound
Poisson--Gamma representation can be written as $Y = sum_(j = 1)^N Z_j$
where the Poisson intensity is

$ nu = frac(mu^(2 - p), phi (2 - p)) $

and the Gamma severities have shape

$ gamma = frac(2 - p, p - 1) $

and rate

$ c = frac(mu^(1 - p), phi (p - 1)) $

These identities are equivalent to the parameter correspondence in Delong,
Lindholm and Wüthrich [1, Proposition 2.1].

= 10. Recovering frequency times severity

== Recovering the pure-premium rate

The mean Gamma severity is $zeta = gamma / c$. Substituting the Tweedie
parameter mapping,

$ zeta = frac(frac(2 - p, p - 1), frac(mu^(1 - p), phi (p - 1))) = phi (2 - p) mu^(p - 1) $

The claim-frequency rate is

$ nu = frac(mu^(2 - p), phi (2 - p)) $

Multiplying frequency and severity,

$ nu zeta = frac(mu^(2 - p), phi (2 - p)) times phi (2 - p) mu^(p - 1) = mu^(2 - p + p - 1) = mu $

Hence

$ mu = nu zeta $

Thus the Tweedie mean is precisely the actuarial pure-premium rate.

= 11. Explicit derivation of the Tweedie variance rate

== Variance rate derivation (part 1)

We now show explicitly why $nu bb("E")[Z^2] = phi mu^p$.

For a Gamma variable with shape $gamma$ and rate $c$,

$ bb("E")[Z^2] = frac(gamma (gamma + 1), c^2) $

From $gamma = frac(2 - p, p - 1)$ we get

$ gamma + 1 = frac(2 - p, p - 1) + 1 = frac(1, p - 1) $

Therefore

$ gamma (gamma + 1) = frac(2 - p, (p - 1)^2) $

== Variance rate derivation (part 2)

Also,

$ c = frac(mu^(1 - p), phi (p - 1)) "  so  " c^2 = frac(mu^(2 - 2p), phi^2 (p - 1)^2) $

Hence

$ bb("E")[Z^2] = (2 - p) phi^2 mu^(2p - 2) $

Now multiply by the frequency rate $nu = frac(mu^(2 - p), phi (2 - p))$:

$ nu bb("E")[Z^2] = frac(mu^(2 - p), phi (2 - p)) times (2 - p) phi^2 mu^(2p - 2) = phi mu^p $

The exponent simplifies to $2 - p + 2p - 2 = p$. Therefore

$ nu bb("E")[Z^2] = phi mu^p $

This is the exact origin of the Tweedie variance function in the compound
Poisson--Gamma representation.

= 12. Introducing arbitrary exposure

== Exposure scaling of mean and variance

Return to policy $i$ with exposure $e_i$. The underlying claim-frequency
rate is $nu_i$. The Poisson mean count over the actual observed exposure is

$ Lambda_i = e_i nu_i $

Thus $N_i ~ op("Poisson")(e_i nu_i)$ and aggregate loss is
$S_i = sum_(j = 1)^(N_i) Z_("ij")$.

From the general compound Poisson result,

$ op("Var")(S_i | X_i, e_i) = e_i nu_i bb("E")[Z_("ij")^2 | X_i] $

From the Tweedie parameter mapping just proved,

$ nu_i bb("E")[Z_("ij")^2 | X_i] = phi mu_i^p $

Therefore

$ op("Var")(S_i | X_i, e_i) = e_i phi mu_i^p $

Together with $bb("E")[S_i | X_i, e_i] = e_i mu_i$, both mean and variance
scale *linearly* with risk exposure.

= 13. From aggregate loss to observed pure premium

== Pure premium and its variance

The observed pure premium (also called loss cost or loss rate) is

$ R_i = frac(S_i, e_i) $

Its conditional mean is

$ bb("E")[R_i | X_i, e_i] = mu_i $

Its conditional variance is

$ op("Var")(R_i | X_i, e_i) = frac(1, e_i^2) op("Var")(S_i | X_i, e_i) = frac(e_i phi mu_i^p, e_i^2) $

Hence

$ op("Var")(R_i | X_i, e_i) = frac(phi, e_i) mu_i^p $

= 14. Exact distribution of the pure premium

== Demonstration by scaling

$ R_i = frac(S_i, e_i) = sum_(j = 1)^(N_i) frac(Z_("ij"), e_i) $

The count remains $N_i ~ op("Poisson")(e_i nu_i)$ and

$ frac(Z_("ij"), e_i) ~ op("Gamma")(gamma, e_i c_i) $

Thus $R_i$ is again compound Poisson--Gamma, with:

- Poisson mean $= e_i nu_i$
- Gamma shape $= gamma$
- Gamma rate $= e_i c_i$

Because compound Poisson--Gamma distributions are exactly Tweedie for
$1 < p < 2$, $R_i$ is Tweedie.

== Demonstration by matching parameters

Suppose $R_i ~ op("Tw")_p (mu_i, d_i)$. The compound Poisson intensity
implied by this Tweedie distribution is $frac(mu_i^(2 - p), d_i (2 - p))$.
Set $d_i = phi / e_i$. Then the implied Poisson intensity is

$ frac(mu_i^(2 - p), (phi / e_i)(2 - p)) = e_i frac(mu_i^(2 - p), phi (2 - p)) = e_i nu_i = Lambda_i $

The Tweedie-implied Gamma rate is also $e_i c_i$. The Gamma shape remains
$gamma = (2 - p) / (p - 1)$. All parameters match exactly, so

$ R_i | X_i, e_i ~ op("Tw")_p (mu_i, frac(phi, e_i)) $

= 15. The aggregate loss is Tweedie too

== Scaling property of Tweedie

The Tweedie family has the scaling property

$ Y ~ op("Tw")_p (m, d) => a Y ~ op("Tw")_p (a m, a^(2 - p) d) $

Set $a = e_i$. Then $S_i = e_i R_i$ has dispersion
$e_i^(2 - p) frac(phi, e_i) = phi e_i^(1 - p)$.

Therefore

$ S_i | X_i, e_i ~ op("Tw")_p (e_i mu_i, phi e_i^(1 - p)) $

Check the variance:

$ op("Var")(S_i) = phi e_i^(1 - p) (e_i mu_i)^p = phi e_i^(1 - p) e_i^p mu_i^p = e_i phi mu_i^p $

This agrees exactly with the compound Poisson calculation.

= 16. Why exposure is the sample weight

== Weight derivation

The pure-premium response satisfies
$R_i ~ op("Tw")_p (mu_i, frac(phi, e_i))$, which is equivalent to

$ op("Var")(R_i | X_i, e_i) = frac(phi mu_i^p, e_i) $

In an exponential dispersion model, an observation weight $w_i$ is
represented through $op("Var")(Y_i | X_i) = frac(phi V(mu_i), w_i)$. With
$V(mu_i) = mu_i^p$ for Tweedie, comparing the two expressions gives

$ w_i = e_i $

Therefore, for an actuarial pure-premium model,

$ y_i = R_i = frac(S_i, e_i), quad "sample weight" = e_i $

This is the statistical justification for

```python
y_pure_premium = total_claim_amount / exposure

model.fit(
    X,
    y_pure_premium,
    sample_weight=exposure,
)
```

when the model is `TweedieRegressor` or a weighted Tweedie objective such as
LightGBM's Tweedie regression. The current scikit-learn insurance example
uses exactly this pure-premium construction and exposure weighting [6].

== Practical implementation

#grid(
  columns: 2,
  gutter: 1em,
  [
    *scikit-learn* \
    ```python
    from sklearn.linear_model \
      import TweedieRegressor

    y = total_claim_amount / exposure

    model = TweedieRegressor(
        power=p,
        link="log",
        alpha=0.0,
    )

    model.fit(
        X, y,
        sample_weight=exposure,
    )
    ```
  ],
  [
    *LightGBM* \
    ```python
    import lightgbm as lgb

    y = total_claim_amount / exposure

    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=p,
    )

    model.fit(
        X, y,
        sample_weight=exposure,
    )
    ```
  ],
)

For actuarial interpretation: $hat(mu)_i$ is the predicted pure-premium rate
and $hat(M)_i = e_i hat(mu)_i$ is the predicted aggregate loss.

= 17. The same result from the Tweedie likelihood

== Likelihood weighting

The variance derivation is sufficient for the quasi-likelihood argument:
quasi-likelihood specifies the conditional mean and variance function,
rather than a full conditional distribution. Hence, once the exposure
scaling of the variance is specified, the corresponding rate weight follows
directly from
$op("Var")(R_i | x_i) = frac(phi, e_i) mu_i^p$.

For a full Tweedie exponential dispersion model, we can also see the weight
directly in the likelihood. Write the Tweedie density schematically as

$ f(r_i; theta_i, phi / e_i, p) = exp{frac(r_i theta_i - kappa_p (theta_i), phi / e_i) + a(r_i, phi / e_i, p)} $

Since $1 / (phi / e_i) = e_i / phi$, the part of the log-likelihood depending
on the mean parameter is $frac(e_i, phi)[r_i theta_i - kappa_p (theta_i)]$.

Each observation contributes proportionally to $e_i$, confirming exposure is
a *precision / likelihood weight*.

Delong, Lindholm and Wüthrich [1, equations (2.2)--(2.5)] formulate the
Tweedie exponential dispersion family directly with exposure $w$ and obtain
$op("Var")(Y) = frac(phi, w) mu^p$. Our notation corresponds to $w = e_i$.

= 18. The same result from Tweedie deviance

== Tweedie deviance identity

For $p != 1, 2$, the Tweedie unit deviance is

$ d_p (y, m) = 2 [frac(y^(2 - p), (1 - p)(2 - p)) - frac(y m^(1 - p), 1 - p) + frac(m^(2 - p), 2 - p)] $

The pure-premium likelihood with dispersion $phi / e_i$ leads, for
estimation of the mean model, to an objective proportional to

$ sum_i e_i d_p (R_i, mu_i) $

Thus the actuarial pure-premium objective is

$ sum_i e_i d_p(frac(S_i, e_i), mu_i) $

This is exactly the form implemented when the pure-premium response is
fitted with $"sample_weight" = e_i$.

= 19. A common misconception

== Common dispersion on aggregate loss

A common conceptual mistake is to write
$S_i ~ op("Tw")_p (e_i mu_i, phi)$ with the *same aggregate dispersion
$phi$ for every exposure*. That model implies

$ op("Var")(S_i) = phi (e_i mu_i)^p = phi e_i^p mu_i^p $

which is not linear in exposure for $p != 1$.

By contrast, the compound Poisson process gives
$op("Var")(S_i) = e_i phi mu_i^p$, so the aggregate Tweedie dispersion
must be $phi_(S, i) = phi e_i^(1 - p)$, not a common constant.

#definition(title: "Correct pure-premium model")[
  $R_i ~ op("Tw")_p (mu_i, phi / e_i)$, hence `sample_weight = e_i`.
]

= 20. What a single Tweedie model assumes about frequency and severity

== Frequency-severity link

A single Tweedie pure-premium model with fixed $p$ and common base dispersion
$phi$ links frequency and severity. Its compound-Poisson representation gives

$ nu_i = frac(mu_i^(2 - p), phi (2 - p)) "  and  " zeta_i = phi (2 - p) mu_i^(p - 1) $

Thus frequency and mean severity are both functions of the same $mu_i$; they
satisfy $mu_i = nu_i zeta_i$ but cannot vary completely independently if
$p$ and $phi$ are fixed.

This is one reason actuarial practice often fits separate Poisson frequency
and Gamma severity GLMs. See Delong, Lindholm and Wüthrich [1] for the
detailed relationship, and Smyth and Jørgensen [5] for dispersion modelling
when a homogeneous Tweedie dispersion is too restrictive.

= 21. Modeling interpretation

=== Five-step summary

#definition(title: "The five modeling steps")[
  + *Claim arrivals*:
    $N_i | X_i, e_i ~ op("Poisson")(e_i nu_i)$. Here $nu_i$ is a
    claim-frequency rate.

  + *Claim severity*:
    $Z_("ij") | X_i ~ op("Gamma")(gamma, c_i)$.
    $zeta_i = gamma / c_i$.

  + *Aggregate cost*:
    $S_i = sum_(j = 1)^(N_i) Z_("ij")$.
    $bb("E")[S_i] = e_i nu_i zeta_i = e_i mu_i$.

  + *Pure premium*:
    $R_i = S_i / e_i$.
    $bb("E")[R_i] = mu_i = nu_i zeta_i$.

  + *Tweedie representation*:
    $R_i ~ op("Tw")_p (mu_i, phi / e_i)$,
    hence $"sample_weight" = e_i$.
]

= 22. Minimal practical implementations

== scikit-learn

```python
from sklearn.linear_model import TweedieRegressor

y = total_claim_amount / exposure

model = TweedieRegressor(
    power=p,
    link="log",
    alpha=0.0,
)

model.fit(
    X,
    y,
    sample_weight=exposure,
)
```

For actuarial interpretation: $hat(mu)_i$ is the predicted pure premium
per unit exposure, and $hat(M)_i = e_i hat(mu)_i$ is the predicted
aggregate loss.

== LightGBM

```python
import lightgbm as lgb

y = total_claim_amount / exposure

model = lgb.LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=p,
)

model.fit(
    X,
    y,
    sample_weight=exposure,
)
```

Again, $hat(mu)_i$ is the predicted pure-premium rate and
$hat(M)_i = e_i hat(mu)_i$ is the predicted aggregate loss.

= 23. Assumptions behind exposure weighting

== Modelling assumptions

The result $op("Var")(R_i) = frac(phi mu_i^p, e_i)$ follows from a stochastic
exposure model. The basic assumptions include:

+ exposure measures claim-generating risk volume
+ conditional claim-arrival intensity is proportional to exposure
+ the risk is sufficiently homogeneous over the exposure interval
+ severity does not mechanically change because the observation window is shorter
+ claim arrivals satisfy the Poisson assumption (the working frequency model)
+ for the exact Tweedie result, severities are Gamma
+ frequency and severity are conditionally independent under the collective-risk construction

These assumptions can be questionable under strong seasonality, endogenous
cancellation, catastrophe dependence, or within-policy coverage changes.

= Appendix A. Poisson counts with exposure offset versus claim-frequency rates

== A.1 Notation

This appendix proves the exact analogue for claim frequency. The result is
simpler than for Tweedie because Poisson corresponds to the boundary power
$p = 1$.

For policy $i$:

- $e_i$: exposure
- $nu_i$: expected claim-frequency rate
- $Lambda_i$: expected claim count over the observed exposure
- $N_i$: observed claim count
- $F_i = N_i / e_i$: observed claim-frequency rate

The connection is $Lambda_i = e_i nu_i$.

Assume a log-link frequency model $log nu_i = X_i^⊤ beta$, so
$nu_i = exp(X_i^⊤ beta)$.

== A.2 Count with a log-exposure offset

The count model is $N_i ~ op("Poisson")(Lambda_i)$ with
$Lambda_i = e_i nu_i$. Taking logs,

$ log Lambda_i = log e_i + X_i^⊤ beta $

Thus $log e_i$ is an *offset*: its coefficient is fixed at one.

The Poisson log-likelihood for observation $i$ is

$ ell_i = N_i log(e_i nu_i) - e_i nu_i - log(N_i !) $

Expanding, the terms $N_i log e_i - log(N_i !)$ do not depend on $beta$.
Therefore, for estimating $beta$,

$ ell_i (beta) equiv N_i log nu_i - e_i nu_i $

Using $N_i = e_i F_i$, this becomes
$ell_i (beta) equiv e_i [F_i log nu_i - nu_i]$ -- already the
weighted-rate formulation.

== A.3 Rate with exposure weight

Define the observed claim-frequency rate $F_i = N_i / e_i$.

Its conditional expectation is
$bb("E")[F_i | X_i, e_i] = nu_i$ and its variance is

$ op("Var")(F_i | X_i, e_i) = frac(nu_i, e_i) $

A weighted Poisson EDF / quasi-likelihood model has
$op("Var")(F_i) = nu_i / w_i$. Thus $w_i = e_i$.

#lemma(title: "Caveat")[
  $F_i = N_i / e_i$ is generally not an integer and should not be
  interpreted literally as a Poisson count. The rate formulation is an
  exponential-dispersion / quasi-likelihood representation whose estimating
  equations and deviance are equivalent to those of the count-with-offset
  formulation.
]

== A.4 Equality of the score equations

For the count formulation,
$ell_i (beta) equiv N_i log nu_i - e_i nu_i$. Under the log link,
$∂ nu_i / ∂ beta = nu_i X_i$, so the score is
$∂ ell_i / ∂ beta = X_i (N_i - e_i nu_i)$.

Summing over observations,

$ U_("count") (beta) = sum_i X_i (N_i - e_i nu_i) $

For the rate formulation with weight $e_i$,

$ U_("rate") (beta) = sum_i e_i X_i (F_i - nu_i) = sum_i X_i (N_i - e_i nu_i) $

Hence $U_("rate") (beta) = U_("count") (beta)$. The two
unpenalized estimators solve exactly the same score equation.

== A.5 Poisson deviance identity

The Poisson unit deviance is

$ d_("P") (y, m) = 2 [y log(y / m) - (y - m)] $

with the convention $0 log 0 = 0$. One verifies directly that

$ d_("P") (N_i, e_i nu_i) = e_i d_("P") (frac(N_i, e_i), nu_i) $

Summing over observations,

$ "Poisson count + offset" log e_i <=> "rate" N_i / e_i " + weight" e_i $

for the unpenalized Poisson deviance / likelihood mean model.

== A.6 Why Poisson is especially simple

=== Boundary case

For Tweedie power $p = 1$,

$ e_i^(2 - p) = e_i^1 = e_i $

Several exposure-scaling identities that differ for $1 < p < 2$ reduce to
the same expression in the Poisson boundary case.

== A.7 Practical formulations

#grid(
  columns: 2,
  gutter: 1em,
  [
    *Count + offset*

    ```text
    response = claim_count
    mean     = exposure * frequency_rate
    link     = log
    offset   = log(exposure)
    ```

    $log bb("E")[N_i] = log e_i + X_i^⊤ beta$
  ],
  [
    *Rate + exposure weight*

    ```text
    response      = claim_count / exposure
    sample_weight = exposure
    link          = log
    ```

    $log bb("E")[F_i] = X_i^⊤ beta$
  ],
)

For an unpenalized Poisson mean model, the two formulations produce the
same estimating equations. With regularization or sample-weight
normalization, one should check the exact implemented objective before
claiming identical numerical fits.

= Appendix B. Compact notation map to Delong, Lindholm and Wüthrich

== Notation map

Delong, Lindholm and Wüthrich [1] use $N ~ op("Poi") (lambda w)$, where
$w$ is exposure and their $lambda$ is the expected claim-frequency rate
relative to exposure.

This note renames those quantities as $w <-> e$, $lambda <-> nu$ and
introduces separately

$ Lambda = e nu $

for the absolute Poisson mean count over the observed exposure.

They define aggregate claims $S = sum_(j = 1)^N Z_j$ and show that the
exposure-normalized quantity $S / w$ has the Tweedie exponential-dispersion
representation with $bb("E")[S / w] = mu$ and
$op("Var")(S / w) = frac(phi, w) mu^p$.

This is exactly our result
$R_i = S_i / e_i ~ op("Tw")_p (mu_i, phi / e_i)$.

= References

== Core actuarial references

#set list(marker: [--], indent: 1em)

+ #strong[[1]] Delong, L., Lindholm, M., & Wüthrich, M. V. (2021). *Making Tweedie's compound Poisson model more accessible.* European Actuarial Journal, *11*, 185--226. doi: `10.1007/s13385-021-00264-3`.

+ #strong[[2]] Jørgensen, B., & de Souza, M. C. P. (1994). *Fitting Tweedie's compound Poisson model to insurance claims data.* Scandinavian Actuarial Journal, *1994*(1), 69--93. doi: `10.1080/03461238.1994.10413930`.

+ #strong[[3]] Wüthrich, M. V. *Non-Life Insurance: Mathematics & Statistics.* SSRN Manuscript ID `2319328`, living lecture notes.

+ #strong[[4]] Ohlsson, E., & Johansson, B. (2010). *Non-Life Insurance Pricing with Generalized Linear Models.* Springer.

+ #strong[[5]] Smyth, G. K., & Jørgensen, B. (2002). *Fitting Tweedie's compound Poisson model to insurance claims data: dispersion modelling.* ASTIN Bulletin, *32*(1), 143--157. doi: `10.2143/AST.32.1.1020`.

== Software reference

+ #strong[[6]] scikit-learn developers. *Tweedie regression on insurance claims*, official scikit-learn example.
