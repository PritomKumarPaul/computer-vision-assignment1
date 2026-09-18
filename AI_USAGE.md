# AI Usage

## Tool

OpenAI Codex was used as a pair-programming assistant. I remained responsible
for the experimental questions, architecture choices, verification,
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
   utilities. I executed the experiments and interpreted whether each proposed
   comparison was scientifically useful.
5. Codex helped organize the final evidence and reproducibility instructions;
   I selected the final model using the validation experiments I ran.

## Verification I performed

I reviewed the generated code, inspected the saved split and dataset audit, ran
the training and evaluation commands, checked the learning curves, and compared
saved metrics before accepting experimental conclusions. I also used model
construction and split tests to catch integration errors.

## Incorrect, ineffective, or questionable suggestion

The initial AI-generated training draft focused on loss and accuracy but
omitted explicit efficiency profiling, even though model scale was part of the
experimental question. I identified this omission and requested measurements
for runtime and network complexity. The implementation was revised to record
total and trainable parameters, parameter storage, estimated MACs/FLOPs,
epoch/total time, throughput, and peak GPU memory. These measurements were
important later: they showed that ConvNeXt-Large gained only one validation
image over ConvNeXt-Tiny while requiring 7.05 times as many parameters and 7.71
times as many MACs.

## Important decisions I made

I chose the experimental sequence: begin with the supplied CNN, respond to its
overfitting with RGB augmentation and regularization, compare CNN families
under one recipe, isolate the effect of pretraining and freezing, and finally
scale the validation-winning ConvNeXt family. I also decided to stop development
after ConvNeXt-Large and preserve the labeled test set for a single final
evaluation.
