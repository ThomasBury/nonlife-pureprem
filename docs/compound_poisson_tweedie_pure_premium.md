# Compound Poisson–Gamma, Tweedie Pure Premium, and Exposure Weights

## Non-life insurance pure premium in a nutshell

This note develops, from first principles, the connection between:

1. a Poisson model for claim counts;
2. a Gamma model for individual claim severities;
3. the compound Poisson–Gamma distribution for aggregate claim cost;
4. the Tweedie distribution with power $1<p<2$;
5. the pure premium (loss cost) as a **rate per unit exposure**;
6. the appearance of `sample_weight = exposure` when fitting a Tweedie pure-premium model;
7. the exact equivalence, in the Poisson case, between:
   - claim counts with a log-exposure offset, and
   - observed claim frequency with exposure as sample weight.

The central principle is simple:

> Exposure scales the **amount of risk observed**.  
> If claim arrivals form a Poisson process, both the expected aggregate claim amount and its variance scale linearly with exposure.

The Tweedie result for pure premium is not introduced as an arbitrary distributional assumption. For $1<p<2$, it can be derived exactly from a compound Poisson model with Gamma severities.

# 1. Why notation matters

A common source of confusion is the symbol $\lambda$.

In many texts,

$$
N \sim \operatorname{Poisson}(\lambda)
$$

uses $\lambda$ for the **Poisson mean**, i.e. the expected number of events over the observation interval.

In insurance texts one also often sees

$$
N_i \sim \operatorname{Poisson}(w_i\lambda_i),
$$

where $\lambda_i$ is now interpreted as a **claim-frequency rate per unit exposure**.

Both conventions are valid, but using the same symbol for an absolute expected count and for a rate easily obscures the role of exposure.

This note deliberately uses two different symbols.

## 1.1 Core notation

For policy or observation $i$:

| Symbol | Meaning | Typical unit |
|---|---|---|
| $X_i$ | rating variables / covariates | -- |
| $e_i>0$ | exposure | policy-years |
| $\nu_i$ | claim-frequency (a **rate**) | claims / policy-year |
| $\Lambda_i$ | expected claim **count** over observed exposure | claims |
| $N_i$ | observed number of claims | claims |
| $Z_{ij}$ | amount of claim $j$ | EUR / claim |
| $\zeta_i$ | expected severity | EUR / claim |
| $S_i$ | observed aggregate claim amount | EUR |
| $\mu_i$ | expected pure-premium **rate** | EUR / policy-year |
| $M_i$ | expected aggregate claim amount | EUR |
| $R_i$ | observed pure premium $S_i/e_i$ | EUR / policy-year |
| $p$ | Tweedie power | dimensionless |
| $\phi$ | base Tweedie dispersion parameter | model-dependent units |

The key identities are

$$
\boxed{\Lambda_i=e_i\nu_i}
$$

and

$$
\boxed{\mu_i=\nu_i\zeta_i}
$$

Consequently,

$$
\boxed{M_i=e_i\mu_i}
$$

These distinguish a **rate** from the corresponding **absolute quantity over the observed exposure**.

## 1.2 Sketch of what we want to discuss

Summary of the main derivation, each step will be derived in the next sections. Start with a frequency rate

$$
\nu_i
$$

Convert it into an expected count over actual exposure:

$$
\Lambda_i=e_i\nu_i
$$

Model claim count:

$$
N_i\sim\operatorname{Poisson}(\Lambda_i)
$$

Let mean severity be

$$
\zeta_i
$$

Then pure premium is

$$
\mu_i=\nu_i\zeta_i
$$

Aggregate loss is

$$
S_i=\sum_{j=1}^{N_i}Z_{ij}
$$

For compound Poisson,

$$
\operatorname{Var}(S_i)
=
e_i\nu_i\mathbb E[Z_i^2]
$$

For the Tweedie compound-Poisson–Gamma parameterization,

$$
\nu_i\mathbb E[Z_i^2]
=
\phi\mu_i^p
$$

Therefore,

$$
\operatorname{Var}(S_i)
=
e_i\phi\mu_i^p
$$

Define observed pure premium

$$
R_i=\frac{S_i}{e_i}
$$

Then

$$
\mathbb E[R_i]
=
\mu_i
$$

and

$$
\operatorname{Var}(R_i)
=
\frac{\phi\mu_i^p}{e_i}
$$

Under the exact Gamma-severity assumption,

$$
\boxed{
R_i
\sim
\operatorname{Tw}_p
\left(
\mu_i,
\frac{\phi}{e_i}
\right)
}
$$

Hence the actuarial Tweedie pure-premium fit is

$$
\boxed{
y_i=\frac{S_i}{e_i},
\qquad
\texttt{sample\_weight}_i=e_i
}
$$

# 2. Frequency: rate versus expected count

Assume claim arrivals follow a Poisson process conditional on the policy characteristics $X_i$.

Let

$$
\nu_i=\nu(X_i)
$$

denote the claim-frequency rate.

If exposure is measured in policy-years, then $\nu_i$ has units

$$
\frac{\text{claims}}{\text{policy-year}}
$$

The expected number of claims during exposure $e_i$ is

$$
\Lambda_i=e_i\nu_i
$$

We model

$$
\boxed{
N_i\mid X_i,e_i
\sim
\operatorname{Poisson}(\Lambda_i)
=
\operatorname{Poisson}(e_i\nu_i)
}
$$

Thus,

$$
\mathbb E[N_i\mid X_i,e_i]
=
e_i\nu_i
$$

and because a Poisson random variable has variance equal to its mean,

$$
\operatorname{Var}(N_i\mid X_i,e_i)
=
e_i\nu_i
$$

## 2.1 Concrete example

Suppose a fixed frequency

$$
\nu_i=0.10
\quad\text{claims per policy-year}
$$

For a full year,

$$
e_i=1
\quad\Longrightarrow\quad
\Lambda_i=0.10
$$

For half a year,

$$
e_i=0.5
\quad\Longrightarrow\quad
\Lambda_i=0.05
$$

For three months,

$$
e_i=0.25
\quad\Longrightarrow\quad
\Lambda_i=0.025
$$

Nothing happens to the underlying annualized claim-frequency rate $\nu_i$. Exposure only converts that rate into the expected count over the actually observed risk period.

This is what phrases such as "unit exposure" mean: one simply evaluates the process at $e_i=1$. It is not a new model.

# 3. Severity model

Let

$$
Z_{i1},Z_{i2},\ldots
$$

be individual claim amounts for policy $i$

Conditional on $X_i$, assume the claim amounts are iid and independent of $N_i$

Write

$$
\zeta_i
=
\mathbb E[Z_{ij}\mid X_i]
$$

for mean severity.

For the exact Tweedie compound-Poisson representation we assume Gamma severities. A Gamma distribution is a reasonable assumption, having a strictly positive support a not too thin tail. 

To avoid ambiguity between Gamma **scale** and **rate** conventions, this note uses a shape-rate parameterization:

$$
\boxed{
Z_{ij}\mid X_i
\sim
\operatorname{Gamma}(\gamma,c_i)
}
$$

where

- $\gamma>0$ is the shape;
- $c_i>0$ is the rate.

Therefore,

$$
\mathbb E[Z_{ij}\mid X_i]
=
\frac{\gamma}{c_i}
=
\zeta_i
$$

$$
\operatorname{Var}(Z_{ij}\mid X_i)
=
\frac{\gamma}{c_i^2}
$$

and

$$
\mathbb E[Z_{ij}^2\mid X_i]
=
\frac{\gamma(\gamma+1)}{c_i^2}
$$

The moment-generating function is

$$
M_Z(t)
=
\left(
\frac{c_i}{c_i-t}
\right)^\gamma,
\qquad t<c_i.
$$

This is the same functional parameterization used in the compound-Poisson derivation of Delong, Lindholm and Wüthrich [1, Section 2.1], although Gamma parameter naming conventions vary across texts.

# 4. Aggregate claim cost: the compound Poisson model

Define aggregate claim cost

$$
\boxed{
S_i
=
\sum_{j=1}^{N_i} Z_{ij}
}
$$

with the usual convention that $S_i=0$ when $N_i=0$.

We have

$$
N_i\mid X_i,e_i
\sim
\operatorname{Poisson}(e_i\nu_i)
$$

and iid Gamma severities.

Therefore $S_i$ has a compound Poisson–Gamma distribution.

This is the classical collective risk model underlying the Tweedie distribution for $1<p<2$; see Jørgensen and de Souza [2] and Delong, Lindholm and Wüthrich [1].

# 5. Mean aggregate claim cost

Condition first on the number of claims.

Given $N_i$,

$$
\mathbb E[S_i\mid N_i,X_i,e_i]
=
N_i\zeta_i
$$

Taking expectation again,

$$
\mathbb E[S_i\mid X_i,e_i]
=
\mathbb E[N_i\mid X_i,e_i]\zeta_i
$$

Since

$$
\mathbb E[N_i\mid X_i,e_i]
=
e_i\nu_i
$$

we obtain

$$
\mathbb E[S_i\mid X_i,e_i]
=
e_i\nu_i\zeta_i
$$

Define the expected pure-premium rate

$$
\boxed{
\mu_i
=
\nu_i\zeta_i
}
$$

Hence

$$
\boxed{
\mathbb E[S_i\mid X_i,e_i]
=
e_i\mu_i
}
$$

This is the actuarial identity

$$
\boxed{
\text{pure premium rate}
=
\text{claim-frequency rate}
\times
\text{mean severity}
}
$$

The dimensional check is useful:

$$
\frac{\text{claims}}{\text{policy-year}}
\times
\frac{\text{EUR}}{\text{claim}}
=
\frac{\text{EUR}}{\text{policy-year}}
$$

Multiplying by exposure gives an absolute expected amount:

$$
\text{policy-years}
\times
\frac{\text{EUR}}{\text{policy-year}}
=
\text{EUR}
$$

# 6. Variance of aggregate claim cost

This is the central stochastic calculation.

Using the law of total variance,

$$
\operatorname{Var}(S_i\mid X_i,e_i)
=
\mathbb E[
\operatorname{Var}(S_i\mid N_i,X_i,e_i)
\mid X_i,e_i
]
+
\operatorname{Var}(
\mathbb E[S_i\mid N_i,X_i,e_i]
\mid X_i,e_i
)
$$

Given $N_i$,

$$
\operatorname{Var}(S_i\mid N_i,X_i,e_i)
=
N_i\operatorname{Var}(Z_{ij}\mid X_i)
$$

so the first term is

$$
\mathbb E[N_i\mid X_i,e_i]
\operatorname{Var}(Z_{ij}\mid X_i)
$$

The second term is

$$
\operatorname{Var}(N_i\zeta_i\mid X_i,e_i)
=
\zeta_i^2
\operatorname{Var}(N_i\mid X_i,e_i)
$$

Because $N_i$ is Poisson,

$$
\mathbb E[N_i\mid X_i,e_i]
=
\operatorname{Var}(N_i\mid X_i,e_i)
=
e_i\nu_i
$$

Therefore,

$$
\operatorname{Var}(S_i\mid X_i,e_i)
=
e_i\nu_i
\left[
\operatorname{Var}(Z_{ij}\mid X_i)
+
\zeta_i^2
\right]
$$

But

$$
\operatorname{Var}(Z_{ij})
+
\left(\mathbb E[Z_{ij}]\right)^2
=
\mathbb E[Z_{ij}^2]
$$

Hence

$$
\boxed{
\operatorname{Var}(S_i\mid X_i,e_i)
=
e_i\nu_i
\mathbb E[Z_{ij}^2\mid X_i]
}
$$

This result is true for a general compound Poisson model; Gamma severity has not yet been needed nor assumed.

The exposure dependence is already explicit:

$$
\boxed{
\operatorname{Var}(S_i\mid X_i,e_i)
\propto e_i
}
$$

That linear scaling follows from the Poisson-process assumption.

# 7. The exact Tweedie connection

For $1<p<2$, the Tweedie distribution is exactly a compound Poisson distribution with Gamma jumps.

This is not merely a similarity of first two moments.

The equivalence is distributional.

Delong, Lindholm and Wüthrich [1, Sections 2.1-2.2, Proposition 2.1] give a particularly clear derivation by comparing moment-generating functions. They start from a compound Poisson–Gamma model with claim count mean equal to exposure times claim frequency and identify the corresponding Tweedie parameters.

Jørgensen and de Souza [2] is a foundational actuarial reference for the same compound-Poisson Tweedie process.

# 8. Mean-dispersion notation for the Tweedie distribution

To keep the derivation transparent, define

$$
Y\sim\operatorname{Tw}_p(m,d)
$$

to mean a Tweedie random variable satisfying

$$
\mathbb E[Y]=m
$$

and

$$
\boxed{
\operatorname{Var}(Y)
=
d\,m^p
}
$$

Here:

- $m$ is the mean;
- $d$ is the actual dispersion attached to this observation;
- $p\in(1,2)$ is the Tweedie power.

Later, for a pure-premium observation with exposure $e_i$, we will obtain

$$
d_i=\frac{\phi}{e_i}
$$

This notation deliberately separates the **observation-specific dispersion** $d_i$ from the common base parameter $\phi$.

# 9. Mapping Tweedie parameters to the compound Poisson–Gamma model

For

$$
Y\sim\operatorname{Tw}_p(\mu,\phi),
\qquad 1<p<2,
$$

the associated compound Poisson–Gamma representation can be written as

$$
Y
=
\sum_{j=1}^{N} Z_j
$$

where the Poisson intensity is

$$
\boxed{
\nu
=
\frac{\mu^{2-p}}
{\phi(2-p)}
}
$$

and the Gamma severities have shape

$$
\boxed{
\gamma
=
\frac{2-p}{p-1}
}
$$

and rate

$$
\boxed{
c
=
\frac{\mu^{1-p}}
{\phi(p-1)}
}
$$

These identities are equivalent to the parameter correspondence in Delong, Lindholm and Wüthrich [1, Proposition 2.1].

We now verify explicitly that this gives the correct pure premium.

# 10. Recovering frequency times severity

The mean Gamma severity is

$$
\zeta
=
\frac{\gamma}{c}
$$

Substituting the Tweedie parameter mapping,

$$
\zeta
=
\frac{
\frac{2-p}{p-1}
}{
\frac{\mu^{1-p}}{\phi(p-1)}
}
$$

Therefore,

$$
\zeta
=
\phi(2-p)\mu^{p-1}
$$

The claim-frequency rate is

$$
\nu
=
\frac{\mu^{2-p}}
{\phi(2-p)}
$$

Multiplying frequency and severity,

$$
\nu\zeta
=
\frac{\mu^{2-p}}
{\phi(2-p)}
\phi(2-p)\mu^{p-1}
$$

The constants cancel:

$$
\nu\zeta
=
\mu^{2-p+p-1}
=
\mu
$$

Hence

$$
\boxed{
\mu
=
\nu\zeta
}
$$

Thus the Tweedie mean is precisely the actuarial pure-premium rate.

# 11. Explicit derivation of the Tweedie variance rate

We now show explicitly why

$$
\nu\,\mathbb E[Z^2]
=
\phi\mu^p
$$

For a Gamma variable with shape $\gamma$ and rate $c$,

$$
\mathbb E[Z^2]
=
\frac{\gamma(\gamma+1)}{c^2}
$$

From

$$
\gamma
=
\frac{2-p}{p-1}
$$

we get

$$
\gamma+1
=
\frac{2-p}{p-1}+1
=
\frac{1}{p-1}
$$

Therefore,

$$
\gamma(\gamma+1)
=
\frac{2-p}{(p-1)^2}
$$

Also,

$$
c
=
\frac{\mu^{1-p}}
{\phi(p-1)}
$$

so

$$
c^2
=
\frac{\mu^{2-2p}}
{\phi^2(p-1)^2}
$$

Hence

$$
\mathbb E[Z^2]
=
\frac{
\frac{2-p}{(p-1)^2}
}{
\frac{\mu^{2-2p}}
{\phi^2(p-1)^2}
}
$$

The $(p-1)^2$ factors cancel:

$$
\mathbb E[Z^2]
=
(2-p)\phi^2\mu^{2p-2}
$$

Now multiply by the frequency rate

$$
\nu
=
\frac{\mu^{2-p}}
{\phi(2-p)}
$$

We obtain

$$
\nu\mathbb E[Z^2]
=
\frac{\mu^{2-p}}
{\phi(2-p)}
(2-p)\phi^2\mu^{2p-2}
$$

Canceling one $\phi$ and $(2-p)$,

$$
\nu\mathbb E[Z^2]
=
\phi
\mu^{2-p+2p-2}
$$

The exponent simplifies to

$$
2-p+2p-2=p
$$

Therefore,

$$
\boxed{
\nu\mathbb E[Z^2]
=
\phi\mu^p
}
$$

This is the exact origin of the Tweedie variance function in the compound Poisson–Gamma representation.

# 12. Introducing arbitrary exposure

Return to policy $i$ with exposure $e_i$

The underlying claim-frequency rate is $\nu_i$

The Poisson mean count over the actual observed exposure is

$$
\Lambda_i=e_i\nu_i
$$

Thus,

$$
N_i
\sim
\operatorname{Poisson}(e_i\nu_i)
$$

Aggregate loss is

$$
S_i
=
\sum_{j=1}^{N_i}Z_{ij}
$$

From the general compound Poisson result,

$$
\operatorname{Var}(S_i\mid X_i,e_i)
=
e_i\nu_i
\mathbb E[Z_{ij}^2\mid X_i]
$$

From the Tweedie parameter mapping just proved,

$$
\nu_i
\mathbb E[Z_{ij}^2\mid X_i]
=
\phi\mu_i^p
$$

Therefore,

$$
\boxed{
\operatorname{Var}(S_i\mid X_i,e_i)
=
e_i\phi\mu_i^p
}
$$

Together with

$$
\mathbb E[S_i\mid X_i,e_i]
=
e_i\mu_i
$$

the aggregate claim cost has

$$
\boxed{
\mathbb E[S_i\mid X_i,e_i]
=
e_i\mu_i,
\qquad
\operatorname{Var}(S_i\mid X_i,e_i)
=
e_i\phi\mu_i^p
}
$$

The important point is that both mean and variance scale **linearly** with risk exposure.

# 13. From aggregate loss to observed pure premium

The observed pure premium (also called loss cost or loss rate) is

$$
\boxed{
R_i
=
\frac{S_i}{e_i}
}
$$

Its conditional mean is

$$
\mathbb E[R_i\mid X_i,e_i]
=
\frac{1}{e_i}
\mathbb E[S_i\mid X_i,e_i]
$$

Therefore,

$$
\boxed{
\mathbb E[R_i\mid X_i,e_i]
=
\mu_i
}
$$

Its conditional variance is

$$
\operatorname{Var}(R_i\mid X_i,e_i)
=
\frac{1}{e_i^2}
\operatorname{Var}(S_i\mid X_i,e_i)
$$

Using

$$
\operatorname{Var}(S_i\mid X_i,e_i)
=
e_i\phi\mu_i^p
$$

we obtain

$$
\operatorname{Var}(R_i\mid X_i,e_i)
=
\frac{e_i\phi\mu_i^p}{e_i^2}
$$

Hence

$$
\boxed{
\operatorname{Var}(R_i\mid X_i,e_i)
=
\frac{\phi}{e_i}\mu_i^p
}
$$

This already identifies the exposure weight.

But we can say something stronger: $R_i$ is not merely a random variable with Tweedie-like first two moments. Under the compound Poisson–Gamma assumptions, it is itself exactly Tweedie distributed.

# 14. Exact distribution of the pure premium

There are two useful demonstrations.

## 14.1 Demonstration by scaling the compound Poisson representation

We have

$$
R_i
=
\frac{S_i}{e_i}
=
\frac{1}{e_i}
\sum_{j=1}^{N_i}Z_{ij}
$$

Therefore,

$$
R_i
=
\sum_{j=1}^{N_i}
\frac{Z_{ij}}{e_i}
$$

The count remains

$$
N_i\sim\operatorname{Poisson}(e_i\nu_i)
$$

If

$$
Z_{ij}
\sim
\operatorname{Gamma}(\gamma,c_i)
$$

in shape-rate notation, then

$$
\frac{Z_{ij}}{e_i}
\sim
\operatorname{Gamma}(\gamma,e_i c_i)
$$

Thus $R_i$ is again compound Poisson–Gamma.

Its compound Poisson parameters are therefore:

$$
\text{Poisson mean}
=
e_i\nu_i
$$

$$
\text{Gamma shape}
=
\gamma
$$

$$
\text{Gamma rate}
=
e_i c_i
$$

Because compound Poisson–Gamma distributions are exactly Tweedie for $1<p<2$, $R_i$ is Tweedie.

## 14.2 Demonstration by matching the Tweedie compound-Poisson parameters

Suppose

$$
R_i
\sim
\operatorname{Tw}_p
\left(
\mu_i,
d_i
\right)
$$

The compound Poisson intensity implied by this Tweedie distribution is

$$
\frac{\mu_i^{2-p}}
{d_i(2-p)}
$$

Set

$$
d_i
=
\frac{\phi}{e_i}
$$

Then the implied Poisson intensity is

$$
\frac{\mu_i^{2-p}}
{\frac{\phi}{e_i}(2-p)}
=
e_i
\frac{\mu_i^{2-p}}
{\phi(2-p)}
$$

But

$$
\nu_i
=
\frac{\mu_i^{2-p}}
{\phi(2-p)}
$$

Hence

$$
\boxed{
\frac{\mu_i^{2-p}}
{(\phi/e_i)(2-p)}
=
e_i\nu_i
=
\Lambda_i
}
$$

The Tweedie-implied Gamma rate is

$$
\frac{\mu_i^{1-p}}
{d_i(p-1)}
=
\frac{\mu_i^{1-p}}
{(\phi/e_i)(p-1)}
=
e_i
\frac{\mu_i^{1-p}}
{\phi(p-1)}
=
e_ic_i
$$

This is exactly the Gamma rate of $Z_{ij}/e_i$.

The Gamma shape remains

$$
\gamma
=
\frac{2-p}{p-1}
$$

Thus the compound Poisson count parameter, Gamma shape and Gamma rate all match exactly.

Therefore,

$$
\boxed{
R_i\mid X_i,e_i
\sim
\operatorname{Tw}_p
\left(
\mu_i,
\frac{\phi}{e_i}
\right)
}
$$

This is an **induced distribution**, not an additional arbitrary assumption.

It follows from:

1. Poisson arrivals with intensity proportional to exposure;
2. iid Gamma severities;
3. independence of frequency and severity;
4. the compound Poisson–Gamma / Tweedie parameter identity.

This is the essential result behind Tweedie pure-premium ratemaking.

# 15. The aggregate loss is Tweedie too, but with a different dispersion

We have

$$
R_i
=
\frac{S_i}{e_i}
$$

and

$$
R_i
\sim
\operatorname{Tw}_p
\left(
\mu_i,
\frac{\phi}{e_i}
\right)
$$

The Tweedie family has the scaling property

$$
Y\sim\operatorname{Tw}_p(m,d)
\quad\Longrightarrow\quad
aY
\sim
\operatorname{Tw}_p
\left(
am,
a^{2-p}d
\right)
$$

Set $a=e_i$

Then

$$
S_i=e_iR_i
$$

has mean

$$
e_i\mu_i
$$

and dispersion

$$
e_i^{2-p}
\frac{\phi}{e_i}
=
\phi e_i^{1-p}
$$

Therefore,

$$
\boxed{
S_i\mid X_i,e_i
\sim
\operatorname{Tw}_p
\left(
e_i\mu_i,
\phi e_i^{1-p}
\right)
}
$$

Check the variance:

$$
\operatorname{Var}(S_i)
=
\phi e_i^{1-p}
(e_i\mu_i)^p.
$$

Therefore,

$$
\operatorname{Var}(S_i)
=
\phi e_i^{1-p}e_i^p\mu_i^p
=
e_i\phi\mu_i^p
$$

This agrees exactly with the compound Poisson calculation.

# 16. Why the pure-premium Tweedie uses exposure as its weight

The pure-premium response satisfies

$$
R_i
\sim
\operatorname{Tw}_p
\left(
\mu_i,
\frac{\phi}{e_i}
\right)
$$

Equivalently,

$$
\operatorname{Var}(R_i\mid X_i,e_i)
=
\frac{\phi\mu_i^p}{e_i}
$$

In an exponential dispersion model, an observation weight $w_i$ is represented through

$$
\operatorname{Var}(Y_i\mid X_i)
=
\frac{\phi V(\mu_i)}{w_i}
$$

For Tweedie,

$$
V(\mu_i)
=
\mu_i^p
$$

Thus

$$
\operatorname{Var}(R_i\mid X_i,e_i)
=
\frac{\phi\mu_i^p}{w_i}
$$

Comparing this with

$$
\operatorname{Var}(R_i\mid X_i,e_i)
=
\frac{\phi\mu_i^p}{e_i}
$$

gives

$$
\boxed{
w_i=e_i
}
$$

Therefore, for an actuarial pure-premium model,

$$
\boxed{
y_i
=
R_i
=
\frac{S_i}{e_i},
\qquad
\text{sample weight}
=
e_i
}
$$

This is the statistical justification for

```python
y_pure_premium = total_claim_amount / exposure

model.fit(
    X,
    y_pure_premium,
    sample_weight=exposure,
)
```

when the model is `TweedieRegressor` or a weighted Tweedie objective such as LightGBM's Tweedie regression.

The current scikit-learn insurance example uses exactly this pure-premium construction and exposure weighting [6].

# 17. The same result from the Tweedie likelihood

The variance derivation is sufficient for the quasi-likelihood argument: quasi-likelihood specifies the conditional mean and variance function, rather than a full conditional distribution. Hence, once the exposure scaling of the variance is specified, the corresponding rate weight follows directly from 

$$
\operatorname{Var}(R_i\mid x_i) = \frac{\phi}{e_i} \mu_i^p
$$

It is not, however, sufficient to prove exact equality of two likelihood or deviance objectives. For that, one must use the Tweedie deviance identity.

For a full Tweedie exponential dispersion model, we can also see the weight directly in the likelihood.

Write the Tweedie density schematically as

$$
f(r_i;\theta_i,\phi/e_i,p)
=
\exp
\left\{
\frac{
r_i\theta_i-\kappa_p(\theta_i)
}{
\phi/e_i
}
+
a(r_i,\phi/e_i,p)
\right\}
$$

Since

$$
\frac{1}{\phi/e_i}
=
\frac{e_i}{\phi}
$$

the part of the log-likelihood depending on the mean parameter is

$$
\frac{e_i}{\phi}
\left[
r_i\theta_i
-
\kappa_p(\theta_i)
\right]
$$

Therefore each observation contributes proportionally to $e_i$.

This is why exposure is a **precision / likelihood weight** for the pure-premium response.

Delong, Lindholm and Wüthrich [1, equations (2.2)-(2.5)] formulate the Tweedie exponential dispersion family directly with exposure $w$ and obtain

$$
\operatorname{Var}(Y)
=
\frac{\phi}{w}\mu^p
$$

Our notation corresponds to

$$
w=e_i
$$

# 18. The same result from Tweedie deviance

For $p\neq1,2$, the Tweedie unit deviance is

$$
d_p(y,m)
=
2
\left[
\frac{y^{2-p}}{(1-p)(2-p)}
-
\frac{ym^{1-p}}{1-p}
+
\frac{m^{2-p}}{2-p}
\right].
$$

The pure-premium likelihood with dispersion $\phi/e_i$ leads, for estimation of the mean model, to an objective proportional to

$$
\sum_i
e_i
d_p(R_i,\mu_i)
$$

Thus the actuarial pure-premium objective is

$$
\boxed{
\sum_i
e_i
d_p
\left(
\frac{S_i}{e_i},
\mu_i
\right).
}
$$

This is exactly the form implemented when the pure-premium response is fitted with

$$
\boxed{\texttt{sample\_weight}=e_i.}
$$


# 19. Why this is not the same as forcing common dispersion on aggregate loss

A common conceptual mistake is to write

$$
S_i
\sim
\operatorname{Tw}_p
(e_i\mu_i,\phi)
$$

with the **same aggregate dispersion $\phi$ for every exposure**.

That model implies

$$
\operatorname{Var}(S_i)
=
\phi(e_i\mu_i)^p
=
\phi e_i^p\mu_i^p
$$

For $p\neq1$, this variance is not linear in exposure.

By contrast, the compound Poisson process gives

$$
\boxed{
\operatorname{Var}(S_i)
=
e_i\phi\mu_i^p
}
$$

The corresponding aggregate Tweedie dispersion must therefore be

$$
\boxed{
\phi_{S,i}
=
\phi e_i^{1-p}
}
$$

not a common constant independent of $e_i$

This distinction explains the otherwise confusing appearance of weights such as $e_i^{2-p}$ in algebraic transformations of a **different** aggregate-loss specification.

For ordinary ratemaking with exposure representing risk volume, the natural compound-Poisson pure-premium model is

$$
\boxed{
R_i
\sim
\operatorname{Tw}_p
\left(
\mu_i,
\frac{\phi}{e_i}
\right),
}
$$

hence

$$
\boxed{
\texttt{sample\_weight}=e_i
}
$$

# 20. What a single Tweedie model assumes about frequency and severity

There is an important restriction hidden inside a single Tweedie pure-premium model with fixed $p$ and common base dispersion $\phi$

The compound-Poisson representation gives

$$
\nu_i
=
\frac{\mu_i^{2-p}}
{\phi(2-p)}
$$

and

$$
\zeta_i
=
\phi(2-p)\mu_i^{p-1}
$$

Thus frequency and mean severity are both functions of the same $\mu_i$, they satisfy

$$
\mu_i=\nu_i\zeta_i,
$$

but they cannot vary completely independently if $p$ and $\phi$ are fixed.

This is one reason actuarial practice often fits separate models:

$$
\text{Poisson frequency GLM}
$$

and

$$
\text{Gamma severity GLM}
$$

Delong, Lindholm and Wüthrich [1] discuss precisely the relationship between the Poisson-Gamma parametrization and the Tweedie parametrization, including conditions under which their regression formulations coincide and reasons why the separate Poisson-Gamma parametrization can be more flexible.

Smyth and Jørgensen [5] further discuss dispersion modeling when a homogeneous Tweedie dispersion is too restrictive.

# 21. Modeling interpretation

The model should be read in the following order.

## 21.1 Claim arrivals

$$
\boxed{
N_i\mid X_i,e_i
\sim
\operatorname{Poisson}(e_i\nu_i)
}
$$

Here $\nu_i$ is a claim-frequency **rate**.

## 21.2 Claim severity

$$
\boxed{
Z_{ij}\mid X_i
\sim
\operatorname{Gamma}(\gamma,c_i)
}
$$

Mean severity is

$$
\boxed{
\zeta_i=\frac{\gamma}{c_i}
}
$$

## 21.3 Aggregate claim cost

$$
\boxed{
S_i
=
\sum_{j=1}^{N_i}Z_{ij}
}
$$

Expected aggregate cost is

$$
\boxed{
\mathbb E[S_i]
=
e_i\nu_i\zeta_i
=
e_i\mu_i.
}
$$

## 21.4 Pure-premium rate

$$
\boxed{
R_i
=
\frac{S_i}{e_i}
}
$$

Expected pure premium is

$$
\boxed{
\mathbb E[R_i]
=
\mu_i
=
\nu_i\zeta_i
}
$$

## 21.5 Tweedie representation

For the compound Poisson–Gamma parameter relation,

$$
\boxed{
R_i
\sim
\operatorname{Tw}_p
\left(
\mu_i,
\frac{\phi}{e_i}
\right),
\qquad
1<p<2
}
$$

Therefore,

$$
\boxed{
\operatorname{Var}(R_i)
=
\frac{\phi\mu_i^p}{e_i}
}
$$

and consequently,

$$
\boxed{
\texttt{sample\_weight}=e_i
}
$$

# 22. Minimal practical implementations

## 22.1 scikit-learn

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

For actuarial interpretation:

$$
\widehat{\mu}_i
=
\text{predicted pure premium per unit exposure}
$$

Expected aggregate loss is then

$$
\widehat{M}_i
=
e_i\widehat{\mu}_i
$$

## 22.2 LightGBM

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

Again,

$$
\widehat{\mu}_i
=
\text{predicted pure-premium rate}
$$

and

$$
\widehat{M}_i
=
e_i\widehat{\mu}_i
$$


# 23. Assumptions behind exposure weighting

The result

$$
\operatorname{Var}(R_i)
=
\frac{\phi\mu_i^p}{e_i}
$$

follows from a stochastic exposure model. The basic assumptions include:

1. exposure measures claim-generating risk volume
2. conditional claim-arrival intensity is proportional to exposure
3. the risk is sufficiently homogeneous over the exposure interval
4. severity does not mechanically change merely because the observation window is shorter
5. claim arrivals satisfy the Poisson assumption, at least as the working frequency model
6. for the exact Tweedie result, severities are Gamma
7. frequency and severity are conditionally independent under the collective-risk construction

These assumptions can be questionable when there is strong seasonality, endogenous cancellation, catastrophe dependence, within-policy changes in coverage, or other forms of non-homogeneous exposure.

# 24. Summary of the main derivation

Start with a frequency rate

$$
\nu_i
$$

Convert it into an expected count over actual exposure:

$$
\boxed{
\Lambda_i=e_i\nu_i
}
$$

Model claim count:

$$
\boxed{
N_i\sim\operatorname{Poisson}(\Lambda_i)
}
$$

Let mean severity be

$$
\zeta_i
$$

Then pure premium is

$$
\boxed{
\mu_i=\nu_i\zeta_i
}
$$

Aggregate loss is

$$
S_i=\sum_{j=1}^{N_i}Z_{ij}
$$

For compound Poisson,

$$
\boxed{
\operatorname{Var}(S_i)
=
e_i\nu_i\mathbb E[Z_i^2]
}
$$

For the Tweedie compound-Poisson–Gamma parameterization,

$$
\boxed{
\nu_i\mathbb E[Z_i^2]
=
\phi\mu_i^p
}
$$

Therefore,

$$
\boxed{
\operatorname{Var}(S_i)
=
e_i\phi\mu_i^p
}
$$

Define observed pure premium

$$
R_i=\frac{S_i}{e_i}
$$

Then

$$
\boxed{
\mathbb E[R_i]
=
\mu_i
}
$$

and

$$
\boxed{
\operatorname{Var}(R_i)
=
\frac{\phi\mu_i^p}{e_i}
}
$$

Under the exact Gamma-severity assumption,

$$
\boxed{
R_i
\sim
\operatorname{Tw}_p
\left(
\mu_i,
\frac{\phi}{e_i}
\right)
}
$$

Hence the actuarial Tweedie pure-premium fit is

$$
\boxed{
y_i=\frac{S_i}{e_i},
\qquad
\texttt{sample\_weight}_i=e_i
}
$$


# Appendix A. Poisson counts with exposure offset versus claim-frequency rates with exposure weight

This appendix proves the exact analogue for claim frequency. The result is simpler than for Tweedie because Poisson corresponds to the boundary power $p=1$.

## A.1 Notation

For policy $i$:

- $e_i$: exposure;
- $\nu_i$: expected claim-frequency rate;
- $\Lambda_i$: expected claim count over the observed exposure;
- $N_i$: observed claim count;
- $F_i=N_i/e_i$: observed claim-frequency rate.

The connection is

$$
\boxed{
\Lambda_i=e_i\nu_i.
}
$$

Assume a log-link frequency model

$$
\boxed{
\log\nu_i
=
X_i^\top\beta.
}
$$

Therefore,

$$
\nu_i
=
\exp(X_i^\top\beta).
$$

---

## A.2 Formulation 1: claim count with a log-exposure offset

The count model is

$$
\boxed{
N_i
\sim
\operatorname{Poisson}(\Lambda_i)
}
$$

with

$$
\Lambda_i
=
e_i\nu_i.
$$

Taking logs,

$$
\log\Lambda_i
=
\log e_i
+
\log\nu_i.
$$

Since

$$
\log\nu_i
=
X_i^\top\beta,
$$

we obtain

$$
\boxed{
\log\Lambda_i
=
\log e_i
+
X_i^\top\beta.
}
$$

Thus $\log e_i$ is an **offset**: its coefficient is fixed at one.

The Poisson log-likelihood for observation $i$ is

$$
\ell_i
=
N_i\log\Lambda_i
-
\Lambda_i
-
\log(N_i!).
$$

Substitute

$$
\Lambda_i
=
e_i\nu_i:
$$

$$
\ell_i
=
N_i\log(e_i\nu_i)
-
e_i\nu_i
-
\log(N_i!).
$$

Expand the logarithm:

$$
\ell_i
=
N_i\log e_i
+
N_i\log\nu_i
-
e_i\nu_i
-
\log(N_i!).
$$

The terms

$$
N_i\log e_i
-
\log(N_i!)
$$

do not depend on $\beta$.

Therefore, for estimating $\beta$,

$$
\boxed{
\ell_i(\beta)
\equiv
N_i\log\nu_i
-
e_i\nu_i.
}
$$

Using

$$
N_i=e_iF_i,
$$

this becomes

$$
\ell_i(\beta)
\equiv
e_i
\left[
F_i\log\nu_i
-
\nu_i
\right].
$$

This is already the weighted-rate formulation.

---

## A.3 Formulation 2: observed frequency with exposure weight

Define the observed claim-frequency rate

$$
\boxed{
F_i
=
\frac{N_i}{e_i}.
}
$$

Its conditional expectation is

$$
\mathbb E[F_i\mid X_i,e_i]
=
\frac{1}{e_i}
\mathbb E[N_i\mid X_i,e_i]
=
\nu_i.
$$

Its variance is

$$
\operatorname{Var}(F_i\mid X_i,e_i)
=
\frac{1}{e_i^2}
\operatorname{Var}(N_i\mid X_i,e_i).
$$

Since

$$
\operatorname{Var}(N_i)
=
e_i\nu_i,
$$

we get

$$
\boxed{
\operatorname{Var}(F_i\mid X_i,e_i)
=
\frac{\nu_i}{e_i}.
}
$$

A weighted Poisson EDF / quasi-likelihood model has

$$
\operatorname{Var}(F_i)
=
\frac{\nu_i}{w_i}.
$$

Thus

$$
\boxed{
w_i=e_i.
}
$$

Important:

> $F_i=N_i/e_i$ is generally not an integer and therefore should not be interpreted literally as a Poisson count.  
> The rate formulation is an exponential-dispersion / quasi-likelihood representation whose estimating equations and deviance are equivalent to those of the count-with-offset formulation.

---

## A.4 Equality of the score equations

For the count formulation,

$$
\ell_i(\beta)
\equiv
N_i\log\nu_i
-
e_i\nu_i.
$$

Under the log link,

$$
\log\nu_i=X_i^\top\beta.
$$

Therefore,

$$
\frac{\partial\nu_i}{\partial\beta}
=
\nu_iX_i.
$$

The score is

$$
\frac{\partial\ell_i}{\partial\beta}
=
X_i
\left(
N_i-e_i\nu_i
\right).
$$

Summing over observations,

$$
\boxed{
U_{\mathrm{count}}(\beta)
=
\sum_i
X_i
\left(
N_i-e_i\nu_i
\right).
}
$$

For the rate formulation with weight $e_i$,

$$
U_{\mathrm{rate}}(\beta)
=
\sum_i
e_iX_i
(F_i-\nu_i).
$$

Since

$$
F_i=\frac{N_i}{e_i},
$$

we obtain

$$
U_{\mathrm{rate}}(\beta)
=
\sum_i
e_iX_i
\left(
\frac{N_i}{e_i}
-
\nu_i
\right).
$$

Hence

$$
U_{\mathrm{rate}}(\beta)
=
\sum_i
X_i
\left(
N_i-e_i\nu_i
\right).
$$

Therefore,

$$
\boxed{
U_{\mathrm{rate}}(\beta)
=
U_{\mathrm{count}}(\beta).
}
$$

The two unpenalized estimators solve exactly the same score equation.

---

## A.5 Exact Poisson deviance identity

The Poisson unit deviance is

$$
d_{\mathrm P}(y,m)
=
2
\left[
y\log\left(\frac{y}{m}\right)
-
(y-m)
\right],
$$

with the usual convention $0\log0=0$.

For the count model,

$$
d_{\mathrm P}
(N_i,e_i\nu_i)
=
2
\left[
N_i
\log
\left(
\frac{N_i}{e_i\nu_i}
\right)
-
(N_i-e_i\nu_i)
\right].
$$

For the rate model,

$$
e_i
d_{\mathrm P}
\left(
\frac{N_i}{e_i},
\nu_i
\right)
$$

equals

$$
2e_i
\left[
\frac{N_i}{e_i}
\log
\left(
\frac{N_i/e_i}{\nu_i}
\right)
-
\left(
\frac{N_i}{e_i}-\nu_i
\right)
\right].
$$

Distribute $e_i$:

$$
=
2
\left[
N_i
\log
\left(
\frac{N_i}{e_i\nu_i}
\right)
-
(N_i-e_i\nu_i)
\right].
$$

Therefore,

$$
\boxed{
d_{\mathrm P}
(N_i,e_i\nu_i)
=
e_i
d_{\mathrm P}
\left(
\frac{N_i}{e_i},
\nu_i
\right).
}
$$

Summing over observations,

$$
\boxed{
\sum_i
d_{\mathrm P}
(N_i,e_i\nu_i)
=
\sum_i
e_i
d_{\mathrm P}
(F_i,\nu_i).
}
$$

Thus:

$$
\boxed{
\text{Poisson count + offset }\log e_i
\quad\Longleftrightarrow\quad
\text{claim-frequency rate }N_i/e_i
\text{ + weight }e_i
}
$$

for the unpenalized Poisson deviance / likelihood mean model.

---

## A.6 Why Poisson is especially simple

For Tweedie power $p=1$,

$$
e_i^{2-p}
=
e_i^{1}
=
e_i.
$$

Thus several exposure-scaling identities that differ for $1<p<2$ collapse to the same expression in the Poisson boundary case.

This is one reason the offset-versus-rate equivalence for claim frequency is less controversial and easier to see.

---

## A.7 Practical formulations

### Count + offset

Conceptually:

```text
response = claim_count
mean     = exposure * frequency_rate
link     = log
offset   = log(exposure)
```

Mathematically,

$$
\log\mathbb E[N_i]
=
\log e_i
+
X_i^\top\beta.
$$

### Rate + exposure weight

Conceptually:

```text
response      = claim_count / exposure
sample_weight = exposure
link          = log
```

Mathematically,

$$
\log\mathbb E[F_i]
=
X_i^\top\beta,
$$

with

$$
\operatorname{Var}(F_i)
=
\frac{\nu_i}{e_i}.
$$

For an unpenalized Poisson mean model, the two formulations produce the same estimating equations.

With regularization, early stopping, software-specific normalization of sample weights, or other algorithmic penalties, one should check the exact implemented objective before claiming identical numerical fits.

---

# Appendix B. Compact notation map to Delong, Lindholm and Wüthrich

Delong, Lindholm and Wüthrich [1] use

$$
N\sim\operatorname{Poi}(\lambda w),
$$

where:

- $w$ is exposure;
- their $\lambda$ is the expected claim-frequency rate relative to exposure.

This note renames those quantities as

$$
w
\longleftrightarrow
e,
$$

$$
\lambda
\longleftrightarrow
\nu,
$$

and introduces separately

$$
\boxed{
\Lambda=e\nu
}
$$

for the absolute Poisson mean count over the observed exposure.

They define aggregate claims

$$
S=\sum_{j=1}^{N}Z_j
$$

and show that the exposure-normalized quantity

$$
\frac{S}{w}
$$

has the Tweedie exponential-dispersion representation with

$$
\mathbb E\left[\frac{S}{w}\right]
=
\mu
$$

and

$$
\operatorname{Var}\left(\frac{S}{w}\right)
=
\frac{\phi}{w}\mu^p.
$$

This is exactly our result

$$
\boxed{
R_i
=
\frac{S_i}{e_i}
\sim
\operatorname{Tw}_p
\left(
\mu_i,
\frac{\phi}{e_i}
\right).
}
$$

# References

## Core actuarial / mathematical references

**[1] Delong, L., Lindholm, M., & Wüthrich, M. V. (2021).**  
*Making Tweedie's compound Poisson model more accessible.*  
European Actuarial Journal, **11**, 185-226.  
doi: `10.1007/s13385-021-00264-3`.

Most directly relevant parts:

- Section 2.1: compound Poisson model with iid Gamma claim sizes;
- Section 2.2: Tweedie's compound Poisson model as an exponential dispersion family;
- equations (2.4)-(2.5): mean and variance with exposure;
- Proposition 2.1: distributional identification between compound Poisson-Gamma and Tweedie;
- Section 3: regression modeling and the relationship between the Poisson-Gamma and Tweedie parametrizations.

This is the clearest single reference for the derivation in this note.

---

**[2] Jørgensen, B., & de Souza, M. C. P. (1994).**  
*Fitting Tweedie's compound Poisson model to insurance claims data.*  
Scandinavian Actuarial Journal, **1994**(1), 69-93.  
doi: `10.1080/03461238.1994.10413930`.

This is a foundational paper for applying the Tweedie compound Poisson process to insurance ratemaking. It explicitly treats data consisting of total claims, claim counts, exposure, a Poisson claim process and Gamma claim sizes, with exposure entering through the dispersion / weight structure.

---

**[3] Wüthrich, M. V.**  
*Non-Life Insurance: Mathematics & Statistics.*  
SSRN Manuscript ID `2319328`, living lecture notes.

The version cited in Delong, Lindholm and Wüthrich [1] is dated January 7, 2020. Delong et al. point to Corollary 7.21 of that version when discussing the exposure-scaled Tweedie compound-Poisson result. Section and corollary numbering can change between versions of the notes.

This is a broad actuarial reference for collective risk models, claim-size distributions, premium calculation and GLMs.

---

**[4] Ohlsson, E., & Johansson, B. (2010).**  
*Non-Life Insurance Pricing with Generalized Linear Models.*  
Springer, Berlin / Heidelberg.  
doi: `10.1007/978-3-642-10791-7`.

This is a practical actuarial GLM reference for claim frequency, pure premium, exposure, tariff modeling and weighted GLMs.

---

**[5] Smyth, G. K., & Jørgensen, B. (2002).**  
*Fitting Tweedie's compound Poisson model to insurance claims data: dispersion modelling.*  
ASTIN Bulletin, **32**(1), 143-157.  
doi: `10.2143/AST.32.1.1020`.

This paper is useful for understanding why modeling Tweedie dispersion may be necessary rather than assuming one homogeneous dispersion parameter across heterogeneous insurance risks.

## Software reference

**[6] scikit-learn developers.**  
*Tweedie regression on insurance claims*, official scikit-learn example.

The example defines pure premium as total claim amount per unit exposure and fits the pure-premium Tweedie regression using exposure as `sample_weight`. It also contrasts the direct Tweedie approach with separate Poisson frequency and Gamma severity models.

# Final takeaway

For standard non-life ratemaking under the compound Poisson–Gamma construction,

$$
\boxed{
N_i
\sim
\operatorname{Poisson}(e_i\nu_i)
}
$$

with frequency rate $\nu_i$, not an ambiguous overloaded Poisson parameter.

Then

$$
\boxed{
\mu_i
=
\nu_i\zeta_i
}
$$

is the expected pure-premium rate and

$$
\boxed{
R_i
=
\frac{S_i}{e_i}
\sim
\operatorname{Tw}_p
\left(
\mu_i,
\frac{\phi}{e_i}
\right).
}
$$

Consequently,

$$
\boxed{
\operatorname{Var}(R_i)
=
\frac{\phi\mu_i^p}{e_i}
}
$$

and the statistically coherent pure-premium fit is

$$
\boxed{
\texttt{sample\_weight}=e_i.
}
$$

For claim frequency, the corresponding Poisson identity is

$$
\boxed{
N_i\text{ with offset }\log e_i
\quad\Longleftrightarrow\quad
\frac{N_i}{e_i}\text{ with sample weight }e_i.
}
$$
