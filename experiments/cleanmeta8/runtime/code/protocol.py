"""Frozen V0 engineering choices. No test-set inputs or adaptive tuning."""
CONFIG = dict(
    version="nsgs-v0-20261106", student_seed=45, teacher_seed=1045,
    sampler_seed=2045, dev_pool_seed=2026110601, dev_reset_seed=2026110602,
    dev_sequences=256, dev_repeats=2, test_seed=922042,
    envs=4096, rollout=24, cycle_updates=100, total_updates=10000,
    teacher_hidden=[64, 32], teacher_lr=1e-3, uniform_mix=.2,
    logit_bound=5., success_window=100, error_ema_alpha=1/32,
    baseline_beta=.9, gradient_clip=1., score_sum_divisor=100.,
    max_teacher_step_kl=.05, test_interval=1000,
    objective="Mean of per-reset simultaneous four-tolerance eligible-time fractions; native failure or fewer than 100 eligible steps => zero. All 512 resets remain in denominator.",
    credit="Conditional one-cycle REINFORCE: reward is next 100 PPO updates' development improvement. Sum log probabilities of ALL real draws of this teacher version, including boundary draws that may affect bootstrap observations. No exposure weighting. Prior-version in-flight episodes are conditioned-on state, never rescored under current teacher. Future-cycle effects are outside this V0 objective.",
    gradient="-(R - prior_EMA_baseline) * sum_draws(log(p_theta(sequence))) / 100. Constant divisor, not random decision count. p=.2/N+.8*softmax(5*tanh(logits/5)). No differentiation through simulator, student or dev evaluation; no offline reuse.",
    missing="Zero is a numeric placeholder only with explicit observed and valid-error masks; unobserved success is missing, not a synthetic 0 or 0.5 outcome.",
)
TRAIN_POOL_SHA = "86f39ce6b71d17d84c47681fe5e44d0fd7efac605ef366c3d8e91916b3eb56db"
TEST_POOL_SHA = "d48b5850285e26b9cec447a0bb4bd6c84493afec5efa0c7725ddcbacbd714afa"
GLOBAL_FIELDS = ["completed_fraction", "lr_over_1e-3", "action_std", "value_loss",
                 "reward_over_200", "feedback_coverage", "mean_train_good_fraction",
                 "mean_train_failure", "development_score", "development_observed"]
DYNAMIC_FIELDS = ["feedback_observed", "valid_error_observed", "strict_success_history",
                  "velocity_rmse_scaled", "yaw_rmse_scaled", "position_rmse_scaled", "orientation_rmse_scaled",
                  "good_fraction_ema", "failure_ema", "feedback_age_scaled", "valid_error_age_scaled",
                  "log_sample_count", "log_complete_count", "log_transition_count", "history_fill"]
