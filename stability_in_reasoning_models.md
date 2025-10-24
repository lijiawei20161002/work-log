# Stability in 1.5B Reasoning Models: A Latent-Space Analysis

**Author:** Jiawei Li
**Repository:** https://github.com/lijiawei20161002/math_model

---

## 1. Research Question

**Core Question:** Why do 1.5B parameter reasoning models succeed on some AIME-2025 problems but fail on others?

**Hypothesis:** Correct solutions correspond to stable, convergent answer distributions with tight latent state clustering, while failures exhibit divergent sampling behavior and high variance in the latent space.

---

## 2. Key Findings

This work demonstrates a **stability-first model of small-model reasoning**: successful reasoning corresponds to low-variance attractors in latent space, whereas failures manifest as high-variance, budget-sensitive trajectories.

### 2.1 Depth and Latent Convergence

**Stable Reasoning Paths:**
- Adding tokens reinforces a convergent attractor in hidden-state space
- Latent vectors remain tightly clustered
- Answer entropy stays low across generation length

**Unstable Reasoning Paths:**
- Additional tokens amplify wandering and divergence
- High sensitivity to generation length (the "overthinking" phenomenon)
- Latent vectors disperse rather than converge

### 2.2 Width and Sampling Variance

**Stable Problems:**
- Multiple rollouts reveal a sharp answer mode
- High top-1 share (consensus)
- Low answer entropy
- Epistemic uncertainty is low

**Unstable Problems:**
- Samples do not concentrate on a single solution
- Entropy remains high across samples
- Width (more samples) exposes instability rather than resolving it

---

## 3. Experimental Setup

### 3.1 Task and Dataset

**Dataset:** AIME-2025
- Chosen because it remains challenging for 1.5B models
- Contrast with MATH-500, where current 1.5B models already perform well

### 3.2 Model Selection

**Model:** DeepScaleR-1.5B-Preview
- Selected based on empirical measurements showing best performance on AIME-2025 among 1.5B models

**Justification:**
- Measurement 1: Current 1.5B models achieve strong performance on MATH-500 (baseline capability)
- Measurement 2: AIME-2025 remains challenging, providing discrimination for analysis

### 3.3 Experimental Design

**Sampling Protocol:**
- 100 samples per question
- max_tokens = 30,720
- Temperature = 1.0
- Varied max_tokens in separate experiments to measure truncation effects

**Analysis Metrics:**

1. **Answer Entropy:**
   - Measures spread of answer distribution across samples
   - Higher entropy → scattered answers (instability)
   - Lower entropy → consistent answers (stability)

2. **Top-1 Share:**
   - Fraction of samples agreeing with the most common answer
   - High share → strong consensus
   - Low share → instability

3. **Hidden State Analysis:**
   - Cosine similarity (tightness) of latent vectors at intermediate tokens
   - PCA for dimensionality reduction and visualization
   - Layer-wise analysis of convergence/divergence

---

## 4. Results

### 4.1 The Stability Gap

**Observation:** A clear dichotomy exists between correct and incorrect problem solutions.

**Correct Problems:**
- Near-zero answer entropy
- High top-1 share (>0.8 typically)
- Tight clustering of hidden states

**Incorrect Problems:**
- High answer entropy across samples
- Low top-1 share
- Dispersed hidden states, especially in deeper layers

**Interpretation:** Success is not merely about finding the right answer, but about converging to a stable attractor in the model's reasoning space.

### 4.2 Latent Space Tightness

**Methodology:**
- Analyzed hidden states at intermediate token positions
- Computed pairwise cosine similarities within sample sets
- Examined layer-wise convergence patterns

**Key Finding:**
- Correct solutions: Hidden states cluster tightly, showing convergence
- Incorrect solutions: Hidden states spread significantly, suggesting instability and lack of convergence
- Effect is most pronounced in deeper layers

### 4.3 Depth vs. Width: Thinking Longer vs. Thinking Wider

Two distinct approaches to improving reasoning accuracy:

1. **Depth (Think Longer):** Increase max_new_tokens (CoT length)
2. **Width (Deliberate More):** Increase number of rollouts per question (e.g., DeepThink [1] achieves 99.9% accuracy on AIME with pass@512)

**Findings on Depth:**
- For DeepScaleR-1.5B-Preview: Longer token limits improve pass@1 accuracy on AIME-2025
- However, "overthinking" phenomenon observed in some cases where more tokens do not help [2]

**Findings on Width:**
- Increasing sample size (k) does not consistently improve pass@k performance
- More samples expose instability rather than resolve it for unstable problems
- Temperature = 1.0 experiments show that width reveals but does not fix fundamental instability

**Interpretation:** Both depth and width have diminishing returns, and their effectiveness depends on whether the problem admits a stable reasoning path within the model's capabilities.

### 4.4 Diverse Chain-of-Thought Paths to Correct Answers

**Observation:** Different models exhibit varying verbosity for the same problem:
- GPT-4o tends to be more verbose
- Claude 3.5 Sonnet (20241022) uses shorter reasoning paths
- Both arrive at the same correct answer

**Open Question:** Do shorter, more efficient reasoning paths always exist for correct answers?
- Potential factors: model architecture, training data, thinking modes
- Connection to token efficiency and computational cost

**Theoretical Context:**
- Adaptive Computation Time (Graves, 2016) [7] pioneered dynamic computational steps
- Models can "think more" in continuous space at test time [8]
- Can be enabled vertically (recurrent architecture) or horizontally (sequential sampling)

---

## 5. Methodology Details

### 5.1 Latent Space Visualization

**2D Projection Pipeline:**

1. **Text Vectorization (line 108):**
   - Each chain-of-thought reasoning path converted to TF-IDF vector
   - Feature dimension: 50

2. **Dimensionality Reduction (line 112):**
   - t-SNE (t-Distributed Stochastic Neighbor Embedding)
   - Reduction: 50D → 2D for visualization

**Rationale:** TF-IDF captures semantic structure of reasoning paths; t-SNE preserves local neighborhood structure for interpretable visualization.

---

## 6. Theoretical Implications

### 6.1 Stability as a Prerequisite for Correctness

The results suggest that **stability precedes correctness**:
- A model must first converge to a stable attractor in latent space
- Only within stable regions can correct reasoning emerge
- Instability (high entropy, dispersed latents) precludes reliable correctness

### 6.2 The Limits of Scale in Test-Time Compute

**Key Insight:** Simply increasing test-time compute (depth or width) does not guarantee improvement for problems outside the model's stable reasoning basin.

**Related Work:**
- Kinetics: Rethinking Test-Time Scaling Laws [3]
- Scaling LLM Test-Time Compute Optimally [4]
- Inverse Scaling in Test-Time Compute [5]
- Fractional Reasoning via Latent Steering Vectors [6]

### 6.3 The Overthinking Phenomenon

**Definition:** Additional tokens sometimes degrade performance rather than improve it.

**Possible Causes:**
- Model drifts away from stable attractor
- Accumulated errors compound
- Lack of self-correction mechanisms in small models

---

## 7. Limitations and Caveats

1. **Model Scope:** Analysis limited to DeepScaleR-1.5B-Preview on AIME-2025
   - Generalization to other models/tasks unclear

2. **Latent Analysis Method:** TF-IDF + t-SNE is interpretable but lossy
   - May not capture full structure of high-dimensional latent space
   - Alternative methods (UMAP, PCA, direct hidden-state analysis) warranted

3. **Causality:** Correlation between stability and correctness does not prove causation
   - Does stability cause correctness, or do both stem from a third factor?

4. **Sample Size:** 100 samples per question provides statistical power
   - But computational cost limits broader exploration of hyperparameter space

5. **Temperature:** Experiments at T=1.0 only
   - Lower temperatures might show different stability patterns

---

## 8. Proposed Next Steps

### 8.1 Immediate Extensions (Low Effort, High Value)

1. **Temperature Sweep:**
   - Test T ∈ {0.0, 0.3, 0.5, 0.7, 1.0, 1.2}
   - Hypothesis: Lower temperatures reduce sampling variance but may reduce coverage of reasoning space

2. **Comparative Model Analysis:**
   - Apply same methodology to Qwen2.5-Math-1.5B, Gemma-2-2B, and other small reasoning models
   - Identify whether stability-correctness relationship is universal or model-specific

3. **Token Budget Sweep:**
   - Systematically vary max_tokens ∈ {1024, 2048, 4096, 8192, 16384, 30720}
   - Identify optimal token budgets for stable vs. unstable problems

4. **Early Stopping Criterion:**
   - Develop a real-time stability metric (e.g., latent variance in last N tokens)
   - Test whether early stopping when instability is detected improves efficiency

### 8.2 Deeper Analysis (Medium Effort)

5. **Layer-Wise Stability Tracking:**
   - Track hidden-state convergence at each transformer layer
   - Identify at which layer(s) instability emerges
   - Hypothesis: Instability may originate in middle layers

6. **Attention Pattern Analysis:**
   - Examine attention weights for stable vs. unstable reasoning paths
   - Do stable paths show more focused attention on key tokens?

7. **Direct Hidden-State Analysis:**
   - Replace TF-IDF with direct analysis of transformer hidden states
   - Use cosine similarity matrices, eigenvalue analysis, or manifold learning

8. **Prompt Engineering for Stability:**
   - Test whether specific prompts (e.g., "think step-by-step carefully") induce more stable reasoning
   - Quantify prompt impact on latent convergence

### 8.3 Novel Directions (High Effort, High Impact)

9. **Steering Toward Stability:**
   - Train lightweight classifiers to predict stability from early tokens
   - Use predictions to steer generation (e.g., via latent steering vectors [6])
   - Goal: Push unstable trajectories toward stable attractors

10. **Adaptive Compute Allocation:**
    - Allocate more tokens to problems with low early-stage stability
    - Allocate fewer tokens to problems with high early-stage stability
    - Test whether adaptive budgets improve overall accuracy-efficiency tradeoff

11. **Synthetic Training Data for Stability:**
    - Generate synthetic CoT data with enforced convergence properties
    - Fine-tune models on this data
    - Test whether stability-aware training improves generalization

12. **Cross-Task Stability Analysis:**
    - Extend analysis to GSM8K, GPQA, MMLU-Pro, etc.
    - Identify whether stability signatures generalize across domains

13. **Mechanistic Interpretability:**
    - Apply circuit analysis or causal tracing to identify specific attention heads or neurons responsible for stability
    - Test interventions (activation patching, ablation) to validate causal mechanisms

14. **Stability-Aware Ensemble Methods:**
    - Weight samples by stability metrics rather than uniform voting
    - Hypothesis: High-stability samples are more likely correct

15. **Theoretical Formalization:**
    - Develop a formal dynamical systems model of reasoning as attractor dynamics
    - Prove (or disprove) conditions under which stability implies correctness

---

## 9. Implementation Roadmap

### Phase 1: Validation and Refinement (2-4 weeks)
- Steps 1-4 (temperature, models, token budgets, early stopping)
- Goal: Validate core findings and establish reproducibility

### Phase 2: Deep Dive (1-2 months)
- Steps 5-8 (layer-wise, attention, hidden states, prompts)
- Goal: Understand mechanisms underlying stability

### Phase 3: Intervention and Application (2-3 months)
- Steps 9-12 (steering, adaptive compute, training, cross-task)
- Goal: Translate insights into actionable improvements

### Phase 4: Theory and Generalization (3-6 months)
- Steps 13-15 (mechanistic interpretability, ensembles, formalization)
- Goal: Develop a general theory of stability in neural reasoning

---

## 10. References

[1] Deep Think with Confidence. https://arxiv.org/abs/2508.15260

[2] Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs. https://arxiv.org/abs/2412.21187

[3] Kinetics: Rethinking Test-Time Scaling Laws. https://arxiv.org/abs/2506.05333

[4] Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. https://arxiv.org/abs/2408.03314

[5] Inverse Scaling in Test-Time Compute. https://arxiv.org/abs/2507.14417

[6] Fractional Reasoning via Latent Steering Vectors Improves Inference Time Compute. https://arxiv.org/abs/2506.15882

[7] Adaptive Computation Time for Recurrent Neural Networks. https://arxiv.org/abs/1603.08983

[8] Why We Think? | Lil'Log. https://lilianweng.github.io/posts/2025-05-01-thinking/

---

## Appendix A: Key Definitions

**Stability:** Low variance in both answer distribution and latent state trajectory across samples.

**Attractor:** A region in latent space toward which reasoning trajectories converge.

**Answer Entropy:** H = -Σ p(a) log p(a), where p(a) is the empirical probability of answer a across samples.

**Top-1 Share:** max_a [count(a) / total_samples], the fraction of samples producing the most common answer.

**Latent Tightness:** Mean pairwise cosine similarity of hidden states at a given token position across samples.

**Overthinking:** The phenomenon where additional reasoning tokens degrade rather than improve performance.

---

## Appendix B: Experimental Checklist

For reproducibility and future extensions, track:

- [ ] Model name and version
- [ ] Task/dataset and split
- [ ] Number of samples per problem
- [ ] max_tokens setting
- [ ] Temperature
- [ ] Top-p / top-k (if applicable)
- [ ] Random seed
- [ ] Hardware (GPU type, VRAM)
- [ ] Inference framework and version
- [ ] Latent extraction method (layer indices, pooling)
- [ ] Dimensionality reduction method and hyperparameters
- [ ] Entropy calculation method (natural log vs. log2)
- [ ] Statistical tests used (if any)

---

**Document Version:** 1.0
**Last Updated:** 2025-10-24
**Status:** Living document—contributions and feedback welcome via GitHub issues.