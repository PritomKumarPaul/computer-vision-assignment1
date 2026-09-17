# AI Usage

## Tool

OpenAI Codex is being used as a pair programmer. The student remains
responsible for the experimental questions, architecture choices, verification,
interpretation, and final report.

## Representative assistance (living log)

1. Codex inspected the assignment handout and starter notebook, summarized the
   required deliverables, and converted them into a reproducible repository
   structure.
2. Codex audited the dataset for class balance, image integrity, dimensions,
   color modes, and exact duplicates. The audit found six duplicate pairs in
   the `LivingRoom` training class and no exact train/test overlap.
3. Codex implemented a deterministic stratified split that keeps exact
   duplicates together, plus shared training and evaluation utilities.

## Verification performed by the student

The student will review generated code, inspect the saved split summary, run
the automated tests, and compare baseline behavior with the supplied notebook
before accepting experimental conclusions.

## Incorrect, ineffective, or questionable suggestion

To be completed with a real example encountered during development. We will
not invent an AI failure after the experiments.

## Important student decision

The student chose to organize the project around three interpretable stages:
the supplied CNN baseline, a deliberately modest CNN trained from scratch, and
an improved pretrained CNN. Final architecture and training decisions remain
subject to the student's review of validation evidence.
