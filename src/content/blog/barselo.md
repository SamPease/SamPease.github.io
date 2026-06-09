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

The system uses historical game data (790+ games spanning Fall 2023 to present) to train Bradley-Terry variants with margin-of-victory extensions. The approach separates individual skill from team effects through batch optimization, uses regularization designed for sparse-data robustness, and validates predictions using both chronological and "new-team" evaluation modes. The result is an interactive static site with player trajectories, searchable databases, and model comparisons.

This project demonstrates the full machine learning pipeline: exploratory data analysis, model development with competing hypotheses, hyperparameter optimization, cross-validation, and deployment. It also illustrates the tension between predictive accuracy and interpretability: a model can be statistically sound while producing rankings that don't pass the "eye test" for practical use.

## View the Interactive Rankings

[Open BARSElo Rankings](/rankings.html)

The rankings page includes player skill trajectories, searchable databases, team comparisons, and visualizations of different model predictions.

## Project Stats

- 790+ games spanning Fall 2023 - present
- 139 unique teams
- 400+ unique players
- ~240 players with <=15 games on a single team (sparse data challenge)
- Tournament data added recently (cross-team matchups, travel team performance)
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

Updated: March 28, 2026

### BT-Normal (New-team mode - current direction)

- Mean negative log-likelihood: 0.6814
- Win prediction accuracy: 53.1%
- Mean Brier score: 0.2442
- Teams scored: 139
- Cross-mode NLL: 0.6580
- Cross-mode Brier: 0.2329
- Cross-mode accuracy: 59.8%
- Cross-mode games scored: 697

### Why BT-VET Was Not the Main Branch (Despite Lower NLL)

BT-VET occasionally posted a slightly better NLL, but hyperparameter optimization often suppressed the veteran-uncertainty behavior after only a few games. In practice, the model became more volatile than the original L2 formulation and gave small-sample anomalies too much weight.

This reinforced the central tradeoff: the model with the best objective score is not always the best ranking system for this use case. If the goal is short-term prediction, extreme rookie estimates can be acceptable. If the goal is robust rankings that reward sustained evidence, stronger priors are often better. My target is the second case: reduce overfitting to anomalies while still letting genuinely strong new players rise when the data supports it.

### Historical Reference Metrics

- BT-Uncert (new-team): mean NLL 0.666, accuracy 54.4%, mean Brier 0.238, cross-mode accuracy 62.2%
- BT-VET (new-team): mean NLL 0.664, accuracy 56.8%, mean Brier 0.237, cross-mode accuracy 62.6%
- BT-MOV baseline (new-team): mean NLL 0.671, accuracy 54.9%, mean Brier 0.240, cross-mode accuracy 62.1%

BT-Normal is now the preferred development direction because it is more stable and philosophically aligned with the goal: requiring substantial evidence before pushing a player far from league average, without hard-coding arbitrary rookie penalties.

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

### Phase 8: BT-Normal (Current)

The latest iteration is BT-Normal. I removed the BT-Uncert uncertainty term because it was not reliably de-ranking under-observed players in a stable way. A fully Bayesian uncertainty treatment may return later, but for now the priority is robust behavior with fewer moving parts.

BT-Normal replaces fixed L2 regularization with a learned Gaussian-scale prior on skill. Instead of penalizing with λΣ(skill²), the loss uses:

- Σ(skill² / (2τ²) + log τ)

This reframes regularization as a normal prior with variance τ² and learns τ directly from data. Conceptually, this is cleaner than learning λ directly, since λ can be pushed toward zero and overfit.

In practice, unconstrained optimization pushed τ toward zero and collapsed skills by exploiting the objective. To prevent this, I added a Gamma-style barrier term on τ²:

- loss += b / τ² + a log(τ)

I am not fully satisfied with choosing a and b as constants for now, but the approach is still principled: it does not force a specific τ value, it only prevents runaway behavior at 0 or infinity. In practice, it does what I wanted: players need more evidence before moving far from the mean, sparse-data players are naturally pulled toward league average, and strong new players can still rise when performance is consistently strong.

## Technical Approach

The project combines data pipeline, statistical modeling, hyperparameter optimization, and interactive visualization.

### Data Pipeline

Data collection is semi-automated. I download HTML pages from the league's scheduling site (LeagueLobster) manually, then scripts extract game results and team rosters:

- `extract_games.py`: Parses HTML for scored games, deduplicates by (datetime, team1, team2), sorts chronologically
- `extract_teams.py`: Extracts team rosters from standings pages
- `resolve_aliases.py`: Interactive tool to handle player name variations (e.g., "Sam" vs "Samantha")
- Output: `Sports Elo - Games.csv` (~790 games) and `Sports Elo - Teams.csv`

### Model Framework

- **Modular base class** (`models/base.py`) with uniform API across all rating systems
- Each model implements: `update()`, `predict_win_prob()`, `expose()`
- **Hyperparameter search** via Optuna (Bayesian optimization, 100-200 trials per model)
- **Configuration-driven** via `unified_config.json` for reproducible experiments

### Current Model Direction: BT-Normal

- Gaussian margin likelihood (σ parameter)
- Adaptive normal-prior regularization with learned scale τ
- Gamma-style barrier on τ to avoid pathological collapse
- New-team mean NLL: 0.6814
- New-team mean accuracy: 53.1%
- Cross-mode accuracy: 59.8%
- Optimizes ~700 games worth of data over 139 unique teams

### Prior Model: BT-Uncert

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
4. **Adaptive prior regularization:** Fixed L2 penalties can be gamed by hyperparameter search. BT-Normal learns a scale parameter τ for a Gaussian prior and stabilizes τ with a Gamma-style barrier term, improving robustness for sparse-data players.
5. **Static site over Dash:** Pre-compute ratings to JSON, serve with GitHub Pages. No backend maintenance, free hosting, instant load times.

## Challenges and Learnings

### Data Sparsity

This is the core technical challenge. About 240 players (~60%) have <=15 games on a single team. I'm trying to separate "this player is good" from "this team was good" with almost no information. The model needs to downweight sparse-data players while still acknowledging real skill differences.

### Attendance Confounds Everything

Teams are fixed per season even if players stop showing up, but I have no attendance data. The model has to treat attendance patterns as part of "skill" because there's no alternative. This is obviously not ideal: a great player who misses half the games shouldn't be rated the same as someone who shows up every week.

### Tournament Data Adds Signal (But New Questions)

Tournament data has been incredibly valuable: it provides cross-team matchups and validates ratings. The league's travel team (composed of top players) dominates tournaments, giving a strong signal that those players are genuinely good. But it raises the central challenge: which players on that dominant team are actually stars, and which are competent players overestimated by association?

### The Overfitting Lesson

TrueSkill taught me to count parameters. 800 parameters for 800 games equals trouble. Now I'm much more careful about model complexity relative to data size.

### The Eye Test vs. Metrics Tension

I can achieve 62-63% prediction accuracy, but the rankings sometimes feel wrong. Unknown players rank surprisingly high. This is the current open problem: finding principled ways to improve ranking quality without just tweaking toward personal biases.
