---
title: "BARSElo: Player Rating System for Recreational Dodgeball"
description: "A data-driven player rating system for recreational dodgeball using Bradley-Terry variants and margin-of-victory modeling."
date: 2026-01-01
tags: ["machine learning", "ranking systems", "sports analytics"]
draft: false
---

# BARSElo: Player Rating System for Recreational Dodgeball

## Overview

BARSElo is a data-driven skill rating system for recreational dodgeball players in the Big Apple Recreational Sports (BARS) league in NYC. The project emerged from a simple question: can you rank individual player skill when win/loss records are heavily determined by randomly-assigned teams? Rather than relying on traditional win-loss records, I built a machine learning model that uses game outcomes and margin of victory to estimate underlying player skill while controlling for team composition.

The system uses historical game data (950+ games spanning Fall 2023 to present) to train Bradley-Terry variants with margin-of-victory extensions. The approach separates individual skill from team effects through batch optimization, uses regularization designed for sparse-data robustness, and validates predictions using both chronological and "new-team" evaluation modes. The result is an interactive static site with player trajectories, searchable databases, and model comparisons.

This project demonstrates the full machine learning pipeline: exploratory data analysis, model development with competing hypotheses, hyperparameter optimization, cross-validation, and deployment. It also illustrates the tension between predictive accuracy and interpretability: a model can be statistically sound while producing rankings that don't pass the "eye test" for practical use.

## View the Interactive Rankings

[Open BARSElo Rankings](/rankings.html)

The rankings page includes player skill trajectories, searchable databases, team comparisons, and visualizations of different model predictions.

## Project Stats

- 950+ games spanning Fall 2023 - present
- 180 unique teams across three contexts (social league, draft league, tournaments/events)
- 500+ unique players
- ~250 players with <=15 games on a single team (sparse data challenge)
- Tournament data (cross-team matchups, travel team performance)
- Manual data collection and curation

## Repository

[View on GitHub](https://github.com/SamPease/BARSElo)

Full source code, model implementations, hyperparameter optimization logs, and data processing scripts.

## Tech Stack

- **Python 3.x**: Primary programming language
- **scipy.optimize (L-BFGS-B)**: Batch optimization for maximum likelihood estimation
- **Optuna**: Bayesian optimization for hyperparameter search (100-200 trials per model)
- **NumPy**: Numerical computation, vectorized probability calculations
- **Pandas**: Data processing and tabular data management
- **BeautifulSoup**: HTML scraping for game and roster data
- **Plotly.js**: Interactive charts and player skill trajectories
- **Vanilla JavaScript**: Client-side interactivity
- **Git**: Version control (~50 commits over 4 months)
- **GitHub Pages**: Static site hosting

## Current Performance

Updated: July 3, 2026

### BT-Team (current production model)

BT-Team is BT-Normal plus three layers developed in sequence: a Laplace-approximated posterior over skills, learned per-context (social/draft/event) noise scales and pool means, and team random effects correlated by roster overlap. It is the first model to win on every evaluation mode at once:

- New-team mean NLL: 0.6572 (vs 0.6814 for BT-Normal)
- New-team accuracy: 57.4%, mean Brier: 0.2337
- Chronological NLL: 0.6318, accuracy 63.1%
- Tournament holdout NLL: 0.6724 (vs 0.6758 for BT-Normal) — a new evaluation mode, see Phase 9
- Team-tier separation (draft teams above social teams in the team ranking): AUC 0.79 vs 0.72 for BT-Normal

### BT-Normal (previous production model)

- New-team mean NLL: 0.6814, accuracy 53.1%, mean Brier 0.2442
- Chronological NLL: 0.6607
- Tournament holdout NLL: 0.6758

### Why BT-VET Was Not the Main Branch (Despite Lower NLL)

BT-VET occasionally posted a slightly better NLL, but hyperparameter optimization often suppressed the veteran-uncertainty behavior after only a few games. In practice, the model became more volatile than the original L2 formulation and gave small-sample anomalies too much weight.

This reinforced the central tradeoff: the model with the best objective score is not always the best ranking system for this use case. If the goal is short-term prediction, extreme rookie estimates can be acceptable. If the goal is robust rankings that reward sustained evidence, stronger priors are often better. My target is the second case: reduce overfitting to anomalies while still letting genuinely strong new players rise when the data supports it.

### Historical Reference Metrics

- BT-Uncert (new-team): mean NLL 0.666, accuracy 54.4%, mean Brier 0.238, cross-mode accuracy 62.2%
- BT-VET (new-team): mean NLL 0.664, accuracy 56.8%, mean Brier 0.237, cross-mode accuracy 62.6%
- BT-MOV baseline (new-team): mean NLL 0.671, accuracy 54.9%, mean Brier 0.240, cross-mode accuracy 62.1%

BT-Normal was the preferred direction for several months because it is stable and philosophically aligned with the goal: requiring substantial evidence before pushing a player far from league average, without hard-coding arbitrary rookie penalties. The current production model, BT-Team, builds three further layers on top of it (Phases 9-13).

## The Journey: From Simple Elo to Bradley-Terry with Margin of Victory

### Phase 1: Starting Simple with Elo

Like many first attempts at skill rating, I started with Elo from chess. Players begin at 1000, team rating is the roster average, and ratings update after each game from expected vs. actual outcomes. I tuned the K-factor to control movement speed.

It worked surprisingly well, around 62% win prediction accuracy. But Elo is reactive: it rewards players on currently winning teams instead of disentangling individual skill from team composition. It also ignores margin of victory, losing the difference between a 5-0 and a 3-2 game.

### Phase 2: The TrueSkill Experiment

TrueSkill seemed perfect for team-based games. It models uncertainty and has strong Bayesian foundations, but it performed worse than Elo. The core issue was overfitting: ~800 parameters for ~800 games gave too much flexibility. Learning two parameters per player (μ and σ) was too many degrees of freedom for this dataset.

The detour was still useful. It taught me Bayesian modeling and clarified the main limitation: sequential update rules are inherently reactive, so Elo and TrueSkill both absorb momentum from current team performance.

### Phase 3: Margin of Victory Struggles

I kept returning to the same question: how should margin of victory be incorporated correctly? I tried treating score differences as "repeated games," adjusting win probabilities by margin, and modifying uncertainty terms. None felt principled. Then I found margins were approximately Gaussian, which changed the direction.

### Phase 4: The Bradley-Terry Pivot

Instead of sequential updates, I optimized all player skills simultaneously over the full game history. That became the core idea behind Bradley-Terry (BT).

**Core approach:** Each player has a latent skill parameter θ (theta). Team skill is the mean of player skills. I maximize the likelihood of observing all game outcomes given these parameters, with L2 regularization to prevent overfitting.

**Margin breakthrough:** I unified the prediction model to use a Gaussian distribution over skill differences:

- When only win/loss is known: use the CDF (cumulative distribution function) to get P(A wins)
- When margin is known: use the PDF (probability density function) centered at the skill difference
- This naturally leverages more information when available without arbitrary hacks

The model uses scipy.optimize to estimate maximum-likelihood skill values across ~700 games, with hyperparameters for regularization strength (λ) and margin noise (σ).

### Phase 4b: The Davidson Tie Parameter Experiment

After settling on margin-of-victory handling, I explored soccer literature, specifically the Davidson tie parameter. Soccer models often include a draw parameter because ties occur more frequently than simple models predict. I tested an analogous parameter for dodgeball.

The answer was no, but in a useful way. With the Gaussian margin formulation, the optimizer consistently pushed the Davidson tie parameter toward zero. That suggested MOV = 0 already captured games effectively tied in skill, where outcomes were mostly noise. Unlike soccer, dodgeball does not appear to have a separate structural draw effect. This removed an unnecessary degree of freedom.

### Phase 5: The Eye Test Problem

BT-MOV beats Elo on predictive metrics, but the rankings do not always pass the eye test. Unexpected players can jump into the top 10. The model predicts well, but does it reflect true skill?

This tension led to dual evaluation modes. Standard chronological prediction lets reactive models look better than they are ("all players on winning teams are good"). New-team mode trains on games before a team's first appearance, then freezes the model and predicts that team's games. It is harder (56.8% vs 62.6% chrono), but it answers the right question: does skill estimation generalize to unseen team compositions?

### Phase 6: Bayesian Uncertainty for Sparse Data

About 240 players (~60%) have played only one season on one team, 15 games or fewer. For those players, I need to separate "this player is good" from "this team was good" with limited information.

The solution was BT-VET (Bradley-Terry with Veteran Uncertainty). It weighted games by player experience, so newer or less-frequent players carried higher uncertainty. Rookie estimates were pulled toward league average, preventing wild swings from lucky or unlucky small samples. It achieved strong new-team performance (NLL 0.664) and matched the intended philosophy: uncertainty should shrink as players accumulate games.

**Next iteration: BT-Uncert** - This refined uncertainty formulation addressed a ranking issue. In Bayesian systems like TrueSkill, a common ranking rule is μ - kσ (conservative estimates), but that felt too arbitrary for this project. BT-Uncert instead ranked players by **average head-to-head win probability**: each player's expected win rate against every other player. This compressed skill and uncertainty into a 1D ranking that directly answers "who would win the most matchups?" The O(n^2) calculation required vectorized NumPy broadcasting.

### Phase 7: TrueSkill Through Time (TTT) Detour

I explored TrueSkill Through Time as a follow-up, hoping a temporal Bayesian structure would handle sparse-data instability more elegantly. I could not get meaningful signal from my data in its current form.

There may still be potential there, but adopting the framework introduced real tradeoffs. Classic TrueSkill had already underperformed, and fitting fully into TTT pulled me away from choices I cared about, especially custom margin-of-victory handling. Since I already had a strong custom optimization framework, this felt more like a framework mismatch than progress.

### Phase 8: BT-Normal

The latest iteration is BT-Normal. I removed the BT-Uncert uncertainty term because it was not reliably de-ranking under-observed players in a stable way. A fully Bayesian uncertainty treatment may return later, but for now the priority is robust behavior with fewer moving parts.

BT-Normal replaces fixed L2 regularization with a learned Gaussian-scale prior on skill. Instead of penalizing with λΣ(skill²), the loss uses:

- Σ(skill² / (2τ²) + log τ)

This reframes regularization as a normal prior with variance τ² and learns τ directly from data. Conceptually, this is cleaner than learning λ directly, since λ can be pushed toward zero and overfit.

In practice, unconstrained optimization pushed τ toward zero and collapsed skills by exploiting the objective. To prevent this, I added a Gamma-style barrier term on τ²:

- loss += b / τ² + a log(τ)

I am not fully satisfied with choosing a and b as constants for now, but the approach is still principled: it does not force a specific τ value, it only prevents runaway behavior at 0 or infinity. In practice, it does what I wanted: players need more evidence before moving far from the mean, sparse-data players are naturally pulled toward league average, and strong new players can still rise when performance is consistently strong.

### Phase 9: Better Yardsticks — Tournament Holdout and Team-Tier Eye Tests

The recurring problem with every model so far: hyperparameter search maximizes NLL, but NLL doesn't measure what I actually care about. I watched Optuna push a tournament-weight parameter to the top of its search range because it helped new-team NLL — while making predictions on actual tournament games *worse*. The optimizer was exploiting the metric, not finding better skill estimates.

Two new yardsticks fixed this. First, a **tournament holdout mode**: train on all non-tournament games, freeze the model, predict the tournament games. Tournaments are the games I trust most — the strongest fields, guaranteed attendance, maximal effort — so this is the closest thing I have to a ground-truth test set. Critically, I never tune hyperparameters on it; the moment you optimize against a test set it stops being one.

Second, a **team-tier eye test** made quantitative. My informal eye test was substantially "do the travel teams top the team rankings?" — travel rosters self-select the strongest players, so if a model can't put them on top, its player estimates are wrong somewhere. I formalized this as rank-AUCs between team tiers (event/draft/social), computed purely from labels already in the data. The refined criterion: the top of the event tier (travel teams) should beat everything, the draft tier should sit above the social tier (both leagues aim for balance, but draft concentrates committed, stronger players), and the event tier as a whole is expectedly wide — one-day charity mixers are not travel teams.

### Phase 10: BT-Laplace — Honest Uncertainty, and a Humbling Measurement

BT-Uncert's uncertainty had been a hand-set formula (σ shrinks with games played). The principled version is to compute the actual posterior: take the Hessian of the loss at the optimum — available in closed form — and invert it. That gives every player a covariance that reflects the *game graph*: a player whose games are all alongside the same teammates has a skill nearly collinear with theirs and gets a wide posterior automatically; a vet with games across many rosters gets a tight one. Predictions integrate this uncertainty (rosters full of unknowns get probabilities near 0.5), and the head-to-head ranking integrates each pair's skill-difference uncertainty, including the covariance.

It improved prediction NLL. It barely moved the rankings — and the reason is the most important measurement this project has produced: posterior σ ranges from 0.9 to 1.5 against a prior scale of 1.5. Even a 230-game veteran's skill is barely half-identified, because each 8v8 team-mean game carries roughly 1/64 of a game of information about any individual, further diluted by roster collinearity. The eye-test problem was never going to be fixed by uncertainty handling; it lives in the skill estimates themselves, which meant changing the likelihood structure.

As a bonus, the Laplace machinery yields a nearly-free diagnostic I should have built months earlier: an influence tool that answers "which games drive this player's rating?" via a one-step leave-one-out approximation (no refits needed). Every "this ranking looks wrong" report can now be turned into a concrete diagnosis in seconds.

### Phase 11: BT-Context — Learning the Difference Between Leagues (and a Spectacular Collapse)

The data comes from three very different sources: the social league (largest, random rosters, most newbies, variable effort), the draft league (captains draft from a stronger, committed pool), and tournaments (strongest fields, maximal effort). I had always resisted hardcoding differences between them. The resolution: label the *context* in the data (a `league` column derived from registration pages and game weekdays — including a fun edge case where the first draft season shared a night with social and is separable only by time of day), and let the model **learn** per-context parameters: an observation-noise scale and a pool mean for each context, anchored on social, everything under shrink-to-zero priors.

The first version collapsed, instructively. I let the model keep learning the prior scale τ (as BT-Normal does) alongside the new context parameters, and the optimizer found a genuinely better optimum — hundreds of nats better — that ranked players almost perfectly by *what fraction of their games were draft games* (correlation 0.998), with individual performance contributing nearly nothing. Chronological NLL looked fine. The ranking was garbage. It is the cleanest demonstration I have of the central thesis: the model with the best objective score is not always the best ranking system. The mechanism was the per-player log(τ) normalizer subsidizing total shrinkage onto the context means once the per-context noise could absorb the outcomes. The fix: τ became a fixed, tuned hyperparameter in this model family.

Fixed, the learned structure is satisfying. The model discovered that tournament games are the *least* noisy (about half the variance of social games) — independently re-deriving the hand-set tournament weight I had been using, from data. It discovered the draft pool sits well above the social pool. And one more refinement came directly from thinking about how selection works: signing up for draft once is weak evidence (newbies wander in), but signing up *again* means you can keep up — so the draft pool prior saturates with tenure. Event rosters are the opposite: being recruited onto a travel team is strong evidence even at a single game, so that prior stays flat.

### Phase 12: Two Hypotheses the Data Rejected

Two attractive ideas died on contact with evidence, and both negative results were worth the trip.

**Student-t margins.** Blowout losses were dragging down good players on bad teams (the influence diagnostic showed a specific vet's rating propped down by 0-7 and 0-5 losses with weak rosters), so heavy-tailed margin likelihoods seemed like the obvious fix. But the residuals — checked both at the fit and out-of-sample with proper per-game standardization — are Gaussian to slightly *thin*-tailed. The best-fit Student-t degenerates to df=100; logistic and Laplace fit clearly worse. My original Gaussian choice was right, and the blowout problem is not about distribution shape: it's about *credit assignment* — mean aggregation charges every roster member 1/8th of a blowout.

**Attendance noise.** With no attendance data, absences are a known confound, and there's an elegant data-free model: a league roster on any given night is a random subset, so game variance should grow with the roster's internal skill spread. I built it (with the variance frozen EM-style so the optimizer couldn't game it) and the data drove every attendance coefficient to zero. The arithmetic explains why: under 8-player mean aggregation, one absent star shifts the expected margin by ~0.3 points against a noise floor of ~2.6. Attendance is real; its effect on team *means* is second-order.

### Phase 13: BT-Team — The System Has a Parameter Now (Current)

Both chronic eye-test complaints — stacked-team newbies rated too high, bad-team vets rated too low — turned out to be the same defect viewed from opposite ends: everything a team does beyond the mean of its players' skills gets smeared onto the players. The fix is a team random effect: every team gets a shrunken effect δ entering the game mean, with a correlated prior δ ~ N(0, τ²_team·K) where K is the Jaccard similarity between rosters. Iterations of the same core group share their effect through the kernel — no team is named anywhere in the code. Our league's dominant travel program is famous for having a *system* that elevates its players, and that system is now literally a fitted parameter (it carries the largest team effect in the league) instead of inflating every member's individual rating.

Rankings use only player skill — δ is deliberately discarded, because you draft the player, not the system. But the *team* ratings on the site now include it: team strength really is more than the average of its members. Prediction handles never-before-seen teams GP-style: a new iteration of a familiar core inherits the kernel-weighted effect of its predecessors, with appropriately widened uncertainty for truly new rosters.

One honest tension remains, and it is now a single knob. The team-effect scale τ_team trades the eye test against tournament NLL almost linearly: larger values let bad-team vets recover and stacked-team riders drop (good), but slightly soften tournament predictions (bad). Optuna, left alone, sets it near zero — NLL simply does not price what the eye test values. I set it by validating against sentinel players and the team-tier AUCs instead. That is a philosophical position as much as a technical one: metrics are guardrails here, not the objective.

## Technical Approach

The project combines data pipeline, statistical modeling, hyperparameter optimization, and interactive visualization.

### Data Pipeline

Data collection is semi-automated. I download HTML pages from the league's scheduling site (LeagueLobster) manually, then scripts extract game results and team rosters:

- `extract_games.py`: Parses HTML for scored games, deduplicates by (datetime, team1, team2), sorts chronologically
- `extract_teams.py`: Extracts team rosters from standings pages, including league context (social/draft/event) from registration page titles
- `backfill_leagues.py`: Labels remaining teams from game weekdays (draft = Sunday, with a time-of-day rule for the one season draft shared a night with social)
- `resolve_aliases.py`: Interactive tool to handle player name variations (e.g., "Sam" vs "Samantha")
- Output: `Sports Elo - Games.csv` (~970 games) and a per-team roster table with league labels

### Model Framework

- **Modular base class** (`models/base.py`) with uniform API across all rating systems
- Each model implements: `update()`, `predict_win_prob()`, `expose()`
- **Hyperparameter search** via Optuna (Bayesian optimization, 100-200 trials per model)
- **Configuration-driven** via `unified_config.json` for reproducible experiments

### Current Production Model: BT-Team

- Everything from BT-Normal (Gaussian margin likelihood, unified PDF/CDF handling of scored and outcome-only games), with τ as a tuned constant rather than learned (see the Phase 11 collapse)
- Laplace-approximated posterior over player skills (closed-form Hessian, full covariance); posterior-predictive win probabilities and uncertainty-integrated head-to-head rankings
- Learned per-context noise scales and pool means for social/draft/event, with tenure-gated draft pool membership
- Team random effects with a roster-overlap (Jaccard) kernel prior; GP-conditional prediction for unseen teams; effects included in team ratings, excluded from player rankings
- New-team mean NLL: 0.6572, tournament holdout NLL: 0.6724, chronological NLL: 0.6318
- Optimizes ~970 games over 180 teams and 539 players

### Prior Models: BT-Normal, BT-Uncert

- Uncertainty term that decreases with games played (alpha parameter)
- Gaussian margin likelihood (σ parameter)
- L2 regularization on skill parameters (l2_lambda)
- New-team mean NLL: 0.666
- Cross-mode accuracy: 62.2%
- **Novel ranking approach:** Instead of the standard Bayesian μ - kσ (which feels arbitrary), players are ranked by average head-to-head win probability, compressing higher-dimensional skill+uncertainty data into a principled 1D ranking. Required vectorized NumPy calculations for O(n^2) efficiency.

### Key Technical Decisions

1. **Batch optimization over incremental updates:** Sequential update rules (Elo, TrueSkill) inherently reward players on teams currently winning. Batch optimization using all data at once better disentangles individual skill from team composition.
2. **Dual evaluation modes:** Chronological validation lets models cheat by being reactive. New-team mode forces the model to generalize to unseen team compositions, which is the actual hard problem.
3. **Gaussian margin likelihood:** The empirical distribution of margins is Gaussian. Using the PDF when margin is known and CDF when only outcome is known extracts more information without arbitrary weighting.
4. **Adaptive prior regularization:** Fixed L2 penalties can be gamed by hyperparameter search. BT-Normal learns a scale parameter τ for a Gaussian prior and stabilizes τ with a Gamma-style barrier term, improving robustness for sparse-data players. (BT-Team later reverted τ to a tuned constant after the learned version collapsed when combined with learnable context parameters.)
5. **Held-out validation the optimizer can't touch:** Tournament games form a test set that hyperparameter search never sees, and quantitative eye tests (team-tier AUCs, sentinel players) validate ranking quality separately from NLL. Several tuning runs were rejected because they improved the objective while degrading these.
6. **Learn context differences, never hardcode them:** Per-league noise and pool means are fitted parameters under shrinkage priors, with only the context *labels* coming from data. The model discovering that tournament games are the cleanest signal — rather than being told — is the difference between a model of the league and a list of my opinions.
7. **Static site over Dash:** Pre-compute ratings to JSON, serve with GitHub Pages. No backend maintenance, free hosting, instant load times.

## Challenges and Learnings

### Data Sparsity

This is the core technical challenge. About 240 players (~60%) have <=15 games on a single team. I'm trying to separate "this player is good" from "this team was good" with almost no information. The model needs to downweight sparse-data players while still acknowledging real skill differences.

### Attendance Confounds Everything (Less Than I Thought)

Teams are fixed per season even if players stop showing up, but I have no attendance data. The model has to treat attendance patterns as part of "skill" because there's no alternative. I eventually built an explicit attendance model (game noise proportional to roster skill spread — a random subset of a homogeneous roster plays like its mean; a star-plus-newbies roster swings) and the data rejected it flat. The arithmetic is why: under 8-player mean aggregation, one absence barely moves the expected margin relative to game noise. Attendance is real, but its effect on team-level outcomes is second-order — the team random effect absorbs what remains of it.

### Tournament Data Adds Signal (But New Questions)

Tournament data has been incredibly valuable: it provides cross-team matchups and validates ratings. The league's travel team (composed of top players) dominates tournaments, giving a strong signal that those players are genuinely good. But it raises the central challenge: which players on that dominant team are actually stars, and which are competent players overestimated by association?

### The Overfitting Lesson

TrueSkill taught me to count parameters. 800 parameters for 800 games equals trouble. Now I'm much more careful about model complexity relative to data size.

### The Eye Test vs. Metrics Tension

I can achieve 62-63% prediction accuracy, but the rankings sometimes feel wrong. Unknown players rank surprisingly high. For most of the project this was the open problem: finding principled ways to improve ranking quality without just tweaking toward personal biases.

The current answer has three parts. First, make the eye test quantitative and data-derived where possible (tournament holdout, team-tier AUCs) so "feels wrong" becomes a number the optimizer didn't produce. Second, use diagnostics (the game-influence tool, posterior uncertainties) to turn every remaining "feels wrong" into a specific mechanistic diagnosis — every eye-test complaint this year traced to an identifiable modeling defect (blowout credit assignment, pool-membership priors), not to mysterious vibes. Third, accept that one or two hyperparameters are legitimately set by judgment rather than by NLL, and be explicit about which ones and why. The optimizer repeatedly demonstrated it will trade away ranking quality for thousandths of a nat; supervising it is part of the job.
