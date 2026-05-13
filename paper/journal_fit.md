# Journal Fit Note

For the current short letter, **Annals of Mathematics and Artificial
Intelligence** is the better fit.

## Recommendation

Choose **Annals of Mathematics and Artificial Intelligence** for the first
version, provided the manuscript stays modest: a geometric split family, a small
algorithmic observation, an exploratory benchmark, and explicit limitations.

The journal describes itself as focusing on mathematical methods applied to AI,
including quantitative, algorithmic, and machine-learning work. That is close
to the contribution here: a mathematically simple modification of tree
induction, with empirical evidence and implementation details.

## Why Not Mathematics First?

**Mathematics** is plausible because its scope includes algorithms, artificial
intelligence and mathematics, machine learning and data mining, probability and
statistics, and computational mathematics. However, a submission there would
likely need to be more theorem-driven: formal approximation results, consistency
questions, stronger complexity analysis, or a more complete mathematical
treatment of the split family.

## Positioning

The safest pitch is:

> Spherical decision trees are a geometric extension of recursive partitioning.
> They are useful when local radial structure is expected, and they recover
> locally linear splits as a limiting case when centers are far from the data.
> Empirically, spherical forests appear competitive for classification, but the
> method is slower and not uniformly better than classical or oblique trees.

## Claims To Avoid

- Do not claim universal superiority over CART or random forests.
- Do not claim statistically significant gains without paired tests or
  confidence intervals.
- Do not claim that far centers solve XOR under greedy induction; the limitation
  is the one-node impurity criterion, not only the candidate-center range.

## Useful Scope Sources

- Annals of Mathematics and Artificial Intelligence:
  https://link.springer.com/journal/10472/aims-and-scope
- Mathematics:
  https://www.mdpi.com/journal/mathematics/about
