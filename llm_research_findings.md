# Low-Compute LLM Research Findings (as of 6 November 2026)

Scope: late-2025 and 2026 workshop/conference papers on LLM inference optimization, pruning, matrix compression, Indic/multilingual languages, and small-model/low-training-compute work.

## Verification notes

- Sources: official ACL Anthology records, ICLR/AAAI/IJCAI official programs, abstracts, and selected full texts.
- This is a curated review, not an exhaustive systematic review and not independent reproduction. All reported numbers are author-reported.
- Workshop vs main vs Findings vs non-archival is labelled for each paper.
- "Training-free" does not mean compute-free: calibration, decompositions, inference, and search still cost resources.
- WMT 2025 is formally a conference (Conference on Machine Translation), not a workshop.
- OpenReview was challenge-blocked during checking; ICLR affiliations/abstracts were verified on ICLR's official site, AAAI affiliation via the organizer program.

---

# Part 1: 40 workshop papers

## A. 2026 workshops: inference, pruning, compression, reasoning

### 1. Skip One Layer? A Multilingual and Multi-Task Study of Single-Layer Skipping in Modern Language Models
- Venue: GroundLM workshop, EMNLP 2026 (Oct 2026). Training-free.
- https://aclanthology.org/2026.groundlm-1.49/
- Skips individual transformer layers across 9 instruction-tuned models, many tasks, and 10 languages (MMLU). Sometimes skipping one layer improves benchmark score while reducing compute; best layer varies by model, task, domain, language; beneficial layers most frequent in the second half. Open problem: choosing the layer with a tiny validation set without overfitting.

### 2. Faithfulness Is Not Free: Auditing Offline KV-Cache Quantization in Retrieval-Augmented Generation
- Venue: GroundLM, EMNLP 2026 (Oct). Inference-only audit.
- https://aclanthology.org/2026.groundlm-1.38/
- Compares BF16/INT8/INT4 stored KV caches (Qwen2.5-7B-Instruct, RGB and HotpotQA). INT8 nearly lossless; INT4 reduces accuracy and degrades faithfulness even when answers stay correct (>90% of faithfulness flips negative); worse under noisy retrieval/greater depth. Caches dequantized before decoding, so this is a storage study, not a runtime speedup.

### 3. How Far Do SAE Features Persist Across LLM Layers? An Empirical Study with an FFN Pruning Application
- Venue: BlackboxNLP, EMNLP 2026 (Oct). No new training; needs a pretrained SAE.
- https://aclanthology.org/2026.blackboxnlp-1.13/
- Tests whether a pretrained SAE's features persist at deeper layers (3 SAE families, 3 ~8B models). Proposes FARN, a training-free per-prompt FFN pruner using SAE decoder columns weighted by active features plus a bias anchor; keeps ~17-20% of FFN neurons per layer, within 1.5 points of an activation oracle on six MCQ benchmarks.

### 4. Demystifying Entropy-based Selection for Chain-of-Thought Compression in Large Reasoning Models
- Venue: BlackboxNLP, EMNLP 2026 (Oct). Inference/analysis.
- https://aclanthology.org/2026.blackboxnlp-1.59/
- Entropy-based CoT step selection shows no advantage over random pruning in all evaluated settings; low-entropy-token advantage on math is due to numeric tokens; activation patching shows information is distributed across the trace. Useful negative evidence: new reasoning-pruning heuristics must beat random and control for numbers/position/length.

### 5. Improving Efficiency of Ternary Large Language Models through Pruning
- Venue: ClimateNLP, EMNLP 2026 (Oct). Post-training pruning.
- https://aclanthology.org/2026.climatenlp-1.5/
- Six ternary models, four sparsity levels, six pruning techniques. Pruning <5% of parameters cuts inference time and energy by up to 21% and memory by up to 13% with minimal accuracy loss; pruning whole attention layers is best for time/energy, whole MLP layers best for memory. Measures real time, energy, memory.

### 6. Budget-Aware Compression Pipeline for Single-GPU LLM Inference: Methods, Trade-offs, and Coupling Effects
- Venue: GroundLM, EMNLP 2026 (Oct). Calibration-based post-training pipeline.
- https://aclanthology.org/2026.groundlm-1.66/
- Studies interactions among layer-wise pruning, weight quantization, and KV-cache compression. Layer pruning makes weight quantization more robust; KV sparsification complements INT8 KV quantization; static vector quantizers conflict with dynamic caching. Compresses a 70B model to ~33 GB, ~57 tokens/s on 10k-token prompts on one A40, accuracy within 5%. Note: ~33 GB artifact vs ~39 GB peak runtime memory; 70B scale is not cheap to reproduce.

### 7. RSCE: Training-Free Residual Stream Encoding for Persistent Context Amortization
- Venue: KnowFM, ACL 2026 (Jul). Training-free with small calibration.
- https://aclanthology.org/2026.knowfm-1.11/
- Mean-pools residual-stream activations of a document into one vector injected at query time (replacing O(|ctx|) prefill with O(1)). Vector-only injection suppresses parametric recall in 4 of 5 architectures; dual-channel design adds a compact explicit fact block. Compared with LongLLMLingua and EHPC at ~99% token reduction on LongBench and RepoBench-C.

### 8. PE-QAT: Parameter-Efficient Quantization-Aware Training for Large Language Models
- Venue: ACL 2026 Student Research Workshop (Jul). Parameter-efficient training.
- https://aclanthology.org/2026.acl-srw.63/
- Freezes the base model, trains LoRA adapters through fake-quantized merged weights (STE), learns per-channel smoothing factors and activation clipping thresholds for per-channel 4-bit weight-activation quantization. Within 0.11 points of full precision on Llama-2-7B zero-shot tasks while training 1.26% of parameters.

### 9. Detecting Overflow in Compressed Token Representations for Retrieval-Augmented Generation
- Venue: EACL 2026 Student Research Workshop (Mar). Lightweight probes.
- https://aclanthology.org/2026.eacl-srw.59/
- Defines "token overflow" in xRAG-style soft compression: when compressed tokens lack information to answer a query. Query-agnostic saturation statistics separate compressed from uncompressed tokens but weakly detect overflow; query+context probes reach 0.72 AUC-ROC on HotpotQA/SQuADv2/TriviaQA, enabling pre-LLM gating.

### 10. Multilingual Chain-of-Thought Compression via Cross-Lingual Distillation
- Venue: MeLLM, ACL 2026 (Jul). SFT + DPO required.
- https://aclanthology.org/2026.mellm-1.8/
- MCD: translation-with-verification aligned dataset, difficulty-aware sampling, SFT plus DPO to shorten reasoning across languages; reduces reasoning length while keeping accuracy, improves low-resource robustness. Shows multilingual CoT shortening is already studied.

### 11. Pruning via Causal Attribution Preserves Reasoning in Large Language Models
- Venue: Logical Reasoning of LLMs workshop, ICLR 2026 (26 Apr). Training-free, calibration needed.
- https://iclr.cc/virtual/2026/10017395
- Masks attention heads on reasoning examples to score importance, then converts head scores to weight-importance scores for projection-matrix pruning.

### 12. Which Heads Matter for Reasoning? RL-Guided KV Cache Compression
- Venue: Latent & Implicit Thinking workshop, ICLR 2026 (27 Apr). RL-based allocation, not training-free.
- https://iclr.cc/virtual/2026/10016679
- RLKV identifies reasoning-critical heads via RL; critical heads keep full caches, others get constant-size caches; reports 20-50% cache reduction near-lossless.

### 13. Rejection Mixing: Fast Semantic Propagation of Mask Tokens for Efficient DLLM Inference
- Venue: Latent & Implicit Thinking, ICLR 2026 (27 Apr). Training-free; diffusion LLMs only.
- https://iclr.cc/virtual/2026/10016678
- ReMix adds an intermediate continuous representation between masked and decoded tokens with a rejection rule that remasks uncertain tokens to reduce inconsistent parallel decoding.

### 14. ConFu: Contemplate the Future for Better Speculative Sampling
- Venue: Latent & Implicit Thinking, ICLR 2026 (27 Apr). Draft-model training required.
- https://iclr.cc/virtual/2026/10016643
- Contemplate tokens and soft prompts give the draft model future information; dynamic MoE and training objectives; reports 8-11% gains over EAGLE-3 in acceptance and speed.

### 15. Alignment-Constrained Dynamic Pruning for LLMs: Identifying and Preserving Alignment-Critical Circuits
- Venue: Fourth Deployable AI Workshop, AAAI 2026 (26 Jan), non-archival oral. Needs activation statistics.
- https://openreview.net/pdf?id=ApPsu5ReUq (program: https://dai-workshop.github.io/dai_2026.html; preprint: https://arxiv.org/html/2511.07482v1)
- Alignment-Aware Probe Pruning: input-dependent channel pruning with stored activation statistics from general/benign/harmful prompts and a risk-aware gate to preserve alignment-critical channels. Efficiency measured by estimated FLOPs, not wall-clock.

## B. Indic and multilingual workshop papers

### 16. Zero-Shot Cross-Lingual Transfer using Prefix-Based Adaptation
- MRL, EMNLP 2025 (Nov). Parameter-efficient training.
- https://aclanthology.org/2025.mrl-main.26/
- Soft prompt tuning, prefix tuning, Llama Adapter for English->35+ languages across 1B-24B models; up to 6% gain over LoRA on Belebele with Llama 3.1 8B; prefix tuning 1.23M trainable parameters. Frozen backbone cost remains.

### 17. Monolingual Adapter Networks for Efficient Cross-Lingual Alignment
- MRL, EMNLP 2025. Lightweight adaptation.
- https://aclanthology.org/2025.mrl-main.24/
- Introduces GitaDB (640 Sanskrit verses with translations into 5 Indic languages and English); shows retrieval degradation especially for Sanskrit; adapters improve embedding alignment without parallel data or full fine-tuning. Narrow, verse-specific domain.

### 18. Alif: Advancing Urdu Large Language Models via Multilingual Synthetic Data Distillation
- MRL, EMNLP 2025. Fine-tuning.
- https://aclanthology.org/2025.mrl-main.19/
- Llama-3.1-8B adapted with Urdu-Instruct synthetic data (reasoning, translation, cultural grounding); outperforms similar-size models on Urdu tasks with <US$100 training budget (excludes data-generation/curation/evaluation costs).

### 19. The Unreasonable Effectiveness of Model Merging for Cross-Lingual Transfer in LLMs
- MRL, EMNLP 2025. Expert training + merging.
- https://aclanthology.org/2025.mrl-main.10/
- Separate language and math experts combined via Layer-Swapping works best across 3 languages and 4 models; removing unhelpful updates post-training often beats pre-freezing. Merging is not compression.

### 20. Pragyaan: Designing and Curating High-Quality Cultural Post-Training Datasets for Indian Languages
- MRL, EMNLP 2025. Dataset/resource.
- https://aclanthology.org/2025.mrl-main.20/
- Pragyaan-IT (22.5K) and Pragyaan-Align (100K) across 10 Indian languages, 13 categories, 56 subcategories, from 57 datasets, via human-in-the-loop translation and synthetic expansion.

### 21. Do Tokenizers Fail on Informal Hindi Expressions? Evidence from Static, Downstream, and Robustness Analyses
- LoResLM, EACL 2026 (29 Mar). Evaluation.
- https://aclanthology.org/2026.loreslm-1.2/
- MuRIL and IndicBERT do better on clean text, but all tested tokenizers degrade severely under romanization (phrase integrity near zero).

### 22. When Multilingual Evaluation Assumptions Fail: Tokenization Effects Across Scripts
- LoResLM, EACL 2026. Controlled training.
- https://aclanthology.org/2026.loreslm-1.4/
- Orthographic Syllable Pair Encoding vs BPE under matched conditions for polarity detection across Devanagari, Perso-Arabic, Latin; up to 27 macro-F1 points difference for Nepali. Classifier setting, not generative-LLM tokenizer replacement.

### 23. Competence Collapse in Code-Mixed Generation: Spectral Evidence and Mechanistic Recovery via Cross-Lingual Activation Steering
- LoResLM, EACL 2026. 50 calibration examples; activation intervention.
- https://aclanthology.org/2026.loreslm-1.3/
- Hinglish prompts degrade instructional quality (9 architectures); Cross-Lingual Activation Steering (CLAS) injects a competence-gap vector into the residual stream and partially recovers utility across 6 open-weight models while preserving code-mixing. Not tested under quantization/pruning.

### 24. Representation-Aware Prompting for Zero-Shot Marathi Text Classification: IPA, Romanization, Repetition
- LoResLM, EACL 2026. Training-free.
- https://aclanthology.org/2026.loreslm-1.37/
- Augments Marathi text with IPA, romanization or repetition; up to 2.6 accuracy points gain on sentiment/hate detection with two instruction-tuned LLMs; best representation is model dependent; combining all is not consistently better; adds prompt length.

### 25. TeluguEval: A Comprehensive Benchmark for Evaluating LLM Capabilities in Telugu
- LoResLM, EACL 2026. Benchmark.
- https://aclanthology.org/2026.loreslm-1.20/
- Translated, human-curated GSM8K, Winogrande, ARC, CaseHOLD, Hendrycks Ethics; 8 models; English-centric models lose 30-40+ points; failures include script sensitivity, option bias, repetition, code-switching.

### 26. Parameter-Efficient Quality Estimation via Frozen Recursive Models
- LoResLM, EACL 2026. Small trainable head on frozen encoder.
- https://aclanthology.org/2026.loreslm-1.52/
- Frozen XLM-R embeddings with TRM-QE: Spearman 0.370 vs 0.369 for fine-tuned variants, trainable parameters 262M -> 7M; Hindi and Tamil beat MonoTransQuest with fewer trainable parameters; external iteration hurts.

### 27. NE-LID: A Fast and Accurate Language Identification System for Northeast Indian Languages
- WILDRE, LREC 2026 (12 May). Non-LLM lightweight model.
- https://aclanthology.org/2026.wildre-1.14/
- Character n-grams + fastText for 11 Northeast Indian languages; 99.09% accuracy on balanced test; ~0.084 ms/prediction. Useful cheap routing/filter baseline; not validated on noisy/code-mixed text.

### 28. TamilTok: Morphologically-Informed Tokenization for Tamil
- DravidianLangTech, ACL 2026 (3 Jul). Tokenizer/resource.
- https://aclanthology.org/2026.dravidianlangtech-1.7/
- TamilMorph (480,000+ morphologically segmented forms) and a tokenizer incorporating morpheme structure; better morphological alignment and downstream results. No end-to-end token-count/latency savings quantified.

### 29. Beyond Benchmark Accuracy: Robustness Evaluation of Hinglish Sentiment Models
- DravidianLangTech, ACL 2026. Evaluation.
- https://aclanthology.org/2026.dravidianlangtech-1.2/
- XLM-R fine-tuned on 25,543 cleaned tweets: 99.7% in-distribution but 42.5% on a 400-example human-validated adversarial set (negation, sarcasm, contrast, subtle, neutral); weak zero-shot transfer to English TweetEval.

### 30. Exploring the Role of Transliteration in In-Context Learning for Low-resource Languages Written in Non-Latin Scripts
- MRL, EMNLP 2025. Training-free prompting.
- https://aclanthology.org/2025.mrl-main.27/
- Original script vs Latin transliteration vs combined prompts; effectiveness varies by task and model size; sequential labeling gains up to 25%.

## C. Late-2025 workshops: small models, simple methods, negative results

### 31. Design and Analysis of few Million Parameter Transformer-based Language Models trained over a few Million Tokens Dataset
- BabyLM, EMNLP 2025. Small-model pretraining.
- https://aclanthology.org/2025.babylm-main.9/
- Models up to 28M parameters on Simple English Wikipedia; best config 8 layers, d=512, 8 heads, FFN 2048, RoPE, RMSNorm, GELU; training loss 0.1; limited generalization.

### 32. Exploring smaller batch sizes for a high-performing BabyLM model architecture
- BabyLM, EMNLP 2025.
- https://aclanthology.org/2025.babylm-main.12/
- Revisits ELC-BERT under constrained resources; removing its training-time advantage largely removes its architectural advantage. Model for equal-compute reproduction papers.

### 33. Sample-Efficient Language Modeling with Linear Attention and Lightweight Enhancements
- BabyLM, EMNLP 2025.
- https://aclanthology.org/2025.babylm-main.14/
- BLaLM: mLSTM token mixer, short convolutions, sliding window attention, Hedgehog feature maps; Muon optimizer stabilizes convergence and reduces perplexity over AdamW.

### 34. Single layer tiny Co4 outpaces GPT-2 and GPT-BERT
- BabyLM, EMNLP 2025.
- https://aclanthology.org/2025.babylm-main.24/
- 8M-parameter, one-layer, two-head Co4 (O(N)) outperforms GPT-2 on 5/7 zero-shot metrics and 6/7 fine-tuning tasks in the BabyLM setting. Good replication candidate; claim is setting-specific.

### 35. Babies Learn to Look Ahead: Multi-Token Prediction in Small LMs
- BabyLM, EMNLP 2025.
- https://aclanthology.org/2025.babylm-main.41/
- MTP curriculum helps even 130M-parameter models under severe data limits. Training objective gain does not itself give inference speedup.

### 36. Not a nuisance but a useful heuristic: Outlier dimensions favor frequent tokens in language models
- BlackboxNLP, EMNLP 2025. Mechanistic analysis.
- https://aclanthology.org/2025.blackboxnlp-1.6/
- Last-layer outlier dimensions implement a frequent-token prediction heuristic; models counterbalance it when inappropriate. Relevant to quantization/outlier clipping.

### 37. Investigating ReLoRA: Effects on the Learning Dynamics of Small Language Models
- BlackboxNLP, EMNLP 2025.
- https://aclanthology.org/2025.blackboxnlp-1.9/
- On 11M-66M parameter models ReLoRA underperforms full-rank training, amplifies rank deficiency and ill-conditioned updates; motivates adaptive/hybrid-rank approaches.

### 38. DeLTa: A Decoding Strategy based on Logit Trajectory Prediction Improves Factuality and Reasoning Ability
- UncertaiNLP, EMNLP 2025. Training-free decoding.
- https://aclanthology.org/2025.uncertainlp-main.26/
- Linear regression over logit trajectories across layers; up to +4.9% TruthfulQA, +8.1% StrategyQA, +7.3% GSM8K.

### 39. Beyond Repetition: Text Simplification and Curriculum Learning for Data-Constrained Pretraining
- BabyLM, EMNLP 2025.
- https://aclanthology.org/2025.babylm-main.19/
- LLM-simplified data and four schedules; small models prefer low-to-high complexity, larger ones interleaved. Teacher-generation cost counts.

### 40. Linguistic Units as Tokens: Intrinsic and Extrinsic Evaluation with BabyLM
- BabyLM, EMNLP 2025.
- https://aclanthology.org/2025.babylm-main.35/
- BPE vs MorPiece, ParadigmFinder, SylliTok; no tokenizer dominates; morphology-aware best intrinsically and helps comprehension tasks.

---

# Part 2: 17 additional conference/Findings papers (baselines, not workshops)

### 41. M-Wanda: Improving One-Shot Pruning for Multilingual LLMs
- EMNLP 2025 main. https://aclanthology.org/2025.emnlp-main.1369/
- Language-aware activation statistics in the pruning criterion; dynamically adjusts layerwise sparsity by cross-lingual importance. Prior art for multilingual pruning.

### 42. Revisiting Pruning vs Quantization for Small Language Models
- Findings of EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.645/
- SparseGPT, Wanda, GPTQ, AWQ on 6 SLMs (0.5-3.8B), 7 languages, 7 tasks; quantization (especially AWQ) beats pruning; LLM trends (Wanda ~ SparseGPT) do not transfer to SLMs.

### 43. FLRC: Fine-grained Low-Rank Compressor for Efficient LLM Inference
- EMNLP 2025 main. https://aclanthology.org/2025.emnlp-main.755/
- Per-layer rank allocation plus progressive low-rank decoding; up to 17% ROUGE-L improvement on summarization versus prior low-rank methods.

### 44. GRASP: Replace Redundant Layers with Adaptive Singular Parameters for Efficient Model Compression
- EMNLP 2025 main. https://aclanthology.org/2025.emnlp-main.1338/
- Gradient-based attribution on a small calibration set to retain sensitivity-aware singular components in place of removed layers; 90% of original performance at 20% compression.

### 45. FISTAPruner: Layer-wise Post-training Pruning for Large Language Models
- EMNLP 2025 main. https://aclanthology.org/2025.emnlp-main.1498/
- LASSO-like convex model solved by FISTA, cumulative error elimination, parallel pruning, 2:4 extension; 50% removal of LLaMA-3-70B keeping 98.6% (unstructured) / 95.6% (2:4) zero-shot performance.

### 46. 1+1>2: A Synergistic Sparse and Low-Rank Compression Method for Large Language Models (SSLC)
- Findings of EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.765/
- Joint low-rank and sparse optimization solved iteratively; no additional training; Qwen2.5 compressed 50% without drop and >=1.63x speedup.

### 47. Saten: Sparse Augmented Tensor Networks for Post-Training Compression of Large Language Models
- Findings of EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.1287/
- Tensor-train compression of LLMs during fine-tuning with sparse augmentation; embedding handling retains rows for most frequent tokens.

### 48. Beyond Fixed-Length Calibration for Post-Training Compression of LLMs
- Findings of EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.1054/
- Calibration sequence length matters; first-token hidden-state behavior differs; proposes state-aware length calibration improving perplexity and zero-shot performance.

### 49. More Tokens, Lower Precision: Towards the Optimal Token-Precision Trade-off in KV Cache Compression
- Findings of EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.429/
- Information-Bottleneck view; "quantized pruning" (more tokens at lower precision) improves long-context performance.

### 50. CopySpec: Accelerating LLMs with Speculative Copy-and-Paste
- EMNLP 2025 main. https://aclanthology.org/2025.emnlp-main.1337/
- Detects repeated sequences and speculatively copies; up to 2.35x on CNN/DM, 3.08x on MT-Redundant, +49% over speculative decoding on MT-Redundant second turn.

### 51. Faster In-Context Learning for LLMs via N-Gram Trie Speculative Decoding
- EMNLP 2025 main. https://aclanthology.org/2025.emnlp-main.911/
- N-gram trie from context for drafting; speedups on summarization, RAG, context QA (Vicuna-7B, Llama2-7B-Chat, Llama3-8B-Instruct).

### 52. Iterative Layer Pruning for Efficient Translation Inference
- WMT 2025 (conference). https://aclanthology.org/2025.wmt-1.78/
- Iterative layer pruning guided by layer importance on Aya-Expanse-8B (Czech->German, English->Egyptian Arabic) with maintained quality.

### 53. JU-NLP: Improving Low-Resource Indic Translation System with Efficient LoRA-Based Adaptation
- WMT 2025 (conference). https://aclanthology.org/2025.wmt-1.95/
- NLLB-200 and IndicTrans2 with LoRA/DoRA (>90% fewer trainable parameters) for Assamese, Manipuri, Mizo, Bodo plus preprocessing and script-aware post-processing.

### 54. FLRQ: Faster LLM Quantization with Flexible Low-Rank Matrix Sketching
- AAAI 2026 main. https://doi.org/10.1609/aaai.v40i26.39283
- Gaussian rank-1 sketching, flexible ranks, alternating low-rank/clipping updates for quantization.

### 55. Budget-Aware LLM Quantization and Low-Rank Correction via Information-Guided Subspace Matrices (BALANCER)
- IJCAI 2026 main. https://ijcai.org/proceedings/2026/475
- Joint bit and correction-rank allocation under a global storage budget including metadata; greedy allocator; Jetson deployment. Main setup includes gradient-based profiling.

### 56. UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs
- ICLR 2026 main. https://proceedings.iclr.cc/paper_files/paper/2026/hash/370fa2e691f57eb319bc263a07dad4a5-Abstract-Conference.html
- Quantization-aware SVD, structured weight sorting (20x faster), state-aware sorting for SSMs, fused RoPE kernel; cloud preparation includes fine-tuning; on-device pruning up to 35%; 4x-5.7x memory reduction, 2.7x-3.4x throughput; accuracy within 5% at 15% pruning.

### 57. UNComp: Can Matrix Entropy Uncover Sparsity? A Compressor Design from an Uncertainty-Aware Perspective
- EMNLP 2025 main. https://aclanthology.org/2025.emnlp-main.209/
- Truncated matrix entropy identifies low-information structure; KV cache reduced to 4.74%, 6% prefill speedup, 6.4x throughput.

---

# Part 3: Matrix-compression prior art (novelty check)

All broad families have direct LLM precedents; the contribution should be a specific objective, selection rule, or deployment result, not a new factorization family.

| Source | Year / venue status | What it rules out |
|---|---|---|
| CURing Large Models: Compression via CUR Decomposition (https://arxiv.org/abs/2501.04211) | 2025 preprint (acceptance not verified) | First CUR weight compression, activation-aware CUR, core-only healing; uses DEIM row/column selection and power-of-two ranks |
| MoDeGPT: Modular Decomposition for LLM Compression (https://arxiv.org/abs/2408.09632) | ICLR 2025 Oral (preprint 2024) | Nystrom on MLP activation correlations, covariance-weighted SVD for V/O, global modulewise allocation; batch-1 GPU latency roughly unchanged (Appendix B.16) |
| Krony-PT: GPT-2 Compressed with Kronecker Products (https://doi.org/10.1109/FLLM67465.2025.11390968) | FLLM 2025 (preprint 2024) | Generic Kronecker FFN compression and Van Loan/pruning-based init; trained several OpenWebText epochs |
| Rethinking the Role of Tensor Decompositions in Post-Training LLM Compression (https://aclanthology.org/2026.emnlp-main.2679/) | EMNLP 2026 main | Systematic TT/Tucker evaluation across dense and MoE; weight reconstruction does not guarantee model quality |
| Hierarchical Sparse Plus Low Rank Compression of LLM (https://arxiv.org/abs/2601.07839) | CODS 2025 per publisher metadata (arXiv comment says CODS-2026) | Magnitude-based sparse part plus hierarchical low-rank residual with RCM permutation; evidence mostly LLaMA attention projections |
| FLRQ (AAAI 2026) | AAAI 2026 | Randomized sketching for low-rank quantization, flexible rank selection |
| BALANCER (IJCAI 2026) | IJCAI 2026 | Joint rank/bit allocation under byte budget |
| UniQL (ICLR 2026) | ICLR 2026 | Combined low-rank + quantization for edge deployment |

Closest-prior-art summary for proposal niches:
- Worst-language activation-covariance-weighted low rank: closest are M-Wanda and MoDeGPT. A possibly overlapping multilingual-MoE calibration submission (on OpenReview; not accessible) was found but not verified (access blocked); novelty NOT cleared.
- Hardware-budgeted rank/bit allocation: most crowded (BALANCER, FLRC, UniQL).
- Rare Indic-token residual protection: SSLC and Saten (token-frequency embedding residuals) overlap; define the narrow comparison carefully.

Deployment caution: smaller factors do not guarantee lower latency; check BALANCER's accounting of factor and scale storage at "4-bit" budgets.

---

# Part 4: Ten low-compute research proposals

(Hypotheses, not confirmed novel contributions; do a final related-work search before submission.)

1. **Script sensitivity under compression.** Compare native, romanized, and code-mixed matched inputs on original vs quantized vs layer-pruned models; measure extra degradation due to compression, control token count and truncation. Basis: papers 21, 23, 30, 41, 42. No backbone training. Venues: LoResLM, MRL/MeLLM, DravidianLangTech, GroundLM.
2. **Coverage-aware calibration selection.** Under fixed calibration-token budget, select examples by script, fragmentation, rare subwords, length, code-mixing; apply to at least two compression methods. Basis: 41, 48. Risk: gains may vanish after controlling length/language balance.
3. **Rare Indic-token residual protection.** Compress embedding/output matrix; spend a tiny byte budget (0.1% / 0.5%) on residual rows chosen by rare-frequency, frequent-frequency, largest error, activation-aware, or random; compare with spending bytes on higher rank. Saten is the closest prior art.
4. **Worst-language-aware low-rank compression.** Candidate objective: min over rank(L)<=r of max over languages l of tr((W-L) C_l (W-L)^T) / (tr(W C_l W^T)+eps), with C_l the uncentered activation second moment. Start with iterative reweighting. Control sequence length, BOS handling, tokenization fertility.
5. **Steering after compression.** Test whether CLAS-style code-mixed vectors survive quantization: no steering vs original vector vs recalibrated vector (10/25/50 examples); do not count English-only output as recovery.
6. **Budget-aware representation selection.** Choose native/romanized/IPA/repetition before a single LLM call using cheap features; compare with fixed, all-representations, and oracle selectors; test on Marathi plus a held-out Indic language.
7. **Language-aware KV-cache budgets for RAG.** Equal-byte comparison of fewer high-precision vs more low-precision tokens across languages; targeted tests for names, numbers, negation, evidence; human validation of faithfulness evaluators for low-resource languages.
8. **Predict which layer to skip from a tiny calibration set.** Nested selection/evaluation protocol; baselines: random layer, fixed late-layer skip, oracle as upper bound; negative result also publishable.
9. **Cost-aware speculative copying.** Gate CopySpec/n-gram-trie proposals with cheap overlap features for Indic editing, repeated chat, transliteration, grounded generation; report acceptance rate, proposal overhead, slowdowns.
10. **Bytes are not latency.** Profile real kernels on one device; test whether latency-constrained rank/bit allocation differs from storage-optimal allocation; separate prefill and batch-1 decode; include buffers/metadata. BALANCER already covers storage-based joint allocation.

## Prioritization (feasibility/differentiation judgment, not acceptance probability)

| Priority | Project | Training | Main difficulty |
|---|---|---|---|
| 1 | Script/code-mixing sensitivity under compression | none | controlled trustworthy evaluations |
| 2 | Budget-aware representation selection | none or tiny router | generalization beyond one model/dataset |
| 3 | Coverage-aware calibration | none | beating matched-budget baselines |
| 4 | Rare-token residual protection | none initially | gains beyond existing selectors |
| 5 | Steering transfer after compression | none | mechanistic access, robust evaluation |
| 6 | Worst-language low-rank | numerical optimization | objective design, efficient implementation |
| 7 | Hardware-constrained rank/bit allocation | little/none | kernel engineering, accurate measurement |

Best first project: Idea 1. Best simple method paper: Idea 6. Best matrix-compression project: Idea 3, then Idea 4.

---

# Part 5: Low-budget experimental plan

1. Two model families around 1B-3B parameters with verified competence in the chosen languages.
2. English plus three Indic languages of diverse scripts/linguistics.
3. Two tasks (one constrained, one generation).
4. Baselines: original, strong PTQ, one structured/layer-pruning baseline, and the proposed change.
5. Three calibration seeds, more than one compression budget; pilot first, then expand.
6. A single 16-24 GB GPU handles many small-model inference experiments; full-precision baselines, long contexts, and decompositions may need more memory.

Report: per-language quality, macro average, worst-language degradation; time to first token, decode latency/throughput, peak memory, preparation time; calibration tokens, GPU-hours, search trials, teacher/API costs; bootstrap intervals; held-out generalization; failure cases.

Pitfalls: unstructured sparsity is not a speedup without kernel support; lower perplexity does not imply preserved downstream behavior; perplexity is not comparable across tokenizers.

---

# Part 6: Workshop targeting

(Verified venue families with relevant recent papers; not claims about currently open calls.)

- LoResLM: low-resource adaptation, evaluation, calibration
- MRL / MeLLM: multilingual transfer, language imbalance, multilingual compression
- DravidianLangTech: Dravidian-language contributions
- WILDRE: Indian-language resources and evaluation
- BlackboxNLP: mechanistic explanations of compression/steering effects
- GroundLM: efficient and faithful inference, RAG compression, deployment analysis
- ClimateNLP: measured energy/resource efficiency
- BabyLM: small-model pretraining and sample efficiency
- AAAI Deployable AI / relevant ICLR workshops (check each edition's scope and archival policy)

Read first: papers 1, 21, 23, 24, 41, 42, 46, 47, 48.

---

# Part 7: Other related items noted

- Findings of the WMT 2025 Shared Task on Model Compression: https://aclanthology.org/2025.wmt-1.25/
- Vicomtech GeLaCo evolutionary compression (compresses Aya Expanse 8B; no Indic): https://aclanthology.org/2025.wmt-1.77/
- ANVITA (NLLB and ByT5; Nyishi, Khasi, Kokborok): https://aclanthology.org/2025.wmt-1.101/
- Language Surgery in Multilingual Large Language Models (MRL 2025): https://aclanthology.org/2025.mrl-main.30/
- Tokenizer-Aware Cross-Lingual Adaptation (EACL 2026 main): https://aclanthology.org/2026.eacl-long.357/
- Towards Efficient FinBERT (FinNLP 2025): https://aclanthology.org/2025.finnlp-2.6/
- EMNLP 2025 efficiency titles seen in the program (not deeply reviewed): TokenSkip, LightThinker, XQuant.

