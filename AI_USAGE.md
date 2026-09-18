# AI Usage

## Tool

OpenAI Codex is being used as a pair programmer. The student remains
responsible for the experimental questions, architecture choices, verification,
interpretation, and final report.

## Representative assistance

1. Codex inspected the assignment handout and starter notebook, summarized the
   required deliverables, and converted them into a reproducible repository
   structure.
2. Codex audited the dataset for class balance, image integrity, dimensions,
   color modes, and exact duplicates. The audit found six duplicate pairs in
   the `LivingRoom` training class and no exact train/test overlap.
3. Codex implemented a deterministic stratified split that keeps exact
   duplicates together, plus shared training and evaluation utilities.
4. Codex helped implement reusable experiment runners, progress/timing
   instrumentation, model-freezing policies, result tables, and plotting
   utilities. The student executed the experiments and interpreted whether
   each proposed comparison was scientifically useful.
5. Codex helped organize the final evidence and reproducibility instructions;
   model selection remained based on the student's validation experiments.

## Verification performed by the student

The student reviewed generated code, inspected the saved split and dataset
audit, ran the training and evaluation commands, checked the learning curves,
and compared saved metrics before accepting experimental conclusions. Model
construction and split tests were also used to catch integration errors.

## Incorrect, ineffective, or questionable suggestion

The shared from-scratch recipe was initially treated as a reasonable way to
compare MobileNetV3-Large with the other CNN families. MobileNet reached
66.8% training accuracy but remained at the 6.25% chance level on validation,
predicting one class for every image. The suggestion was ineffective for this
architecture. The result was retained and diagnosed rather than discarded;
later experiments used ImageNet pretraining and architecture-appropriate
fine-tuning scopes.

## Important student decision

The student chose the experimental sequence: begin with the supplied CNN,
respond to its overfitting with RGB augmentation and regularization, compare
CNN families under one recipe, isolate the effect of pretraining and freezing,
and finally scale the validation-winning ConvNeXt family. The student also
decided to stop development after ConvNeXt-Large and preserve the labeled test
set for a single final evaluation.
