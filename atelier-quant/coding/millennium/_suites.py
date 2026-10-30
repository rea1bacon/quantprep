"""Répartition des tests de tests_source/ entre les questions du site : un module tests_<clé>.py par question."""
from _harnais import Suite

C = "test_complements.py"
J1, J2, J3, J4, B = "test_day1.py", "test_day2.py", "test_day3.py", "test_day4.py", "test_bonus_data_cleaning.py"
R, D, N = "test_real_exam_recall.py", "test_desk_utilities_exam.py", "test_numpy_drills.py"


class SuiteDouble(Suite):
    """Une question dont les tests viennent de deux fichiers : la suite d'origine, puis les compléments."""

    def __init__(self, fichier, correction, visibles, autres, complements):
        super().__init__(fichier, correction, visibles, autres)
        self.complement = Suite(C, correction, [], complements)

    def evaluer(self, mode, espace):
        r = super().evaluer(mode, espace)
        return r if mode == "visibles" else r + self.complement.evaluer("caches", espace)

    def nombres(self):
        v, t = super().nombres()
        return v, t + self.complement.nombres()[1]


SUITES = {
    "mm1_e1": Suite(J1, "mm_1.py", ["test_quote_is_valid", "test_zero_vol_still_has_positive_spread"],
                    ["test_spread_widens_with_volatility", "test_deterministic_pure_function", "test_spread_is_capped_under_extreme_vol",
                     "test_quote_brackets_ref_price", "test_backtest_widens_spread_during_stress_regime",
                     "test_backtest_produces_realistic_fill_activity", "test_pnl_and_inventory_stay_finite"]),
    "mm1_e2": Suite(C, "mm_1.py", ["test_gm_cas_symetrique"],
                    ["test_gm_sans_informes_ecart_nul", "test_gm_tous_informes", "test_gm_a_priori_asymetrique",
                     "test_gm_valeur_certaine", "test_gm_parametres_invalides"]),
    "mm2_e1": Suite(C, "mm_2.py", ["test_indifference_valeurs"], ["test_indifference_stock_nul", "test_indifference_ecart_entre_achat_et_vente"]),
    "mm2_e2": SuiteDouble(J2, "mm_2.py", ["test_quote_is_valid", "test_zero_inventory_has_no_skew"],
                          ["test_skews_down_when_long", "test_skews_up_when_short", "test_skew_magnitude_grows_with_inventory",
                           "test_extreme_inventory_never_produces_an_invalid_quote", "test_spread_still_widens_with_volatility",
                           "test_backtest_reduces_inventory_variance_vs_naive_fixed_spread", "test_pnl_stays_finite"],
                          ["test_as_formule_exacte", "test_as_decalage_plafonne", "test_as_fin_de_journee_sans_risque"]),
    "mm3_e1": SuiteDouble(J3, "mm_3.py", ["test_hedge_ratio_recovers_true_beta", "test_rebalance_returns_none_inside_no_trade_band"],
                          ["test_hedge_ratio_handles_a_different_planted_beta", "test_rebalance_returns_trade_outside_no_trade_band",
                           "test_rebalance_direction_is_buy_when_reducing_short", "test_rebalance_rejects_non_positive_price",
                           "test_rolling_hedge_reduces_return_variance_substantially"],
                          ["test_beta_egale_pente_des_moindres_carres", "test_ajustement_nul_sans_ordre", "test_bande_atteinte_exactement"]),
    "mm3_e2": Suite(C, "mm_3.py", ["test_couverts_forme"], ["test_couverts_premiere_valeur", "test_couverts_sans_regard_vers_le_futur", "test_couverts_reduit_la_variance"]),
    "mm4_e1": SuiteDouble(J4, "mm_4.py", ["test_half_life_recovers_planted_value"], [], ["test_demi_vie_exacte", "test_demi_vie_infinie", "test_demi_vie_nulle"]),
    "mm4_e2": SuiteDouble(J4, "mm_4.py", ["test_synthetic_price_is_weighted_sum", "test_zscore_matches_hand_computation"],
                          ["test_signal_no_trade_when_zscore_small", "test_signal_no_trade_when_edge_below_commission_floor",
                           "test_signal_sell_etf_when_rich", "test_signal_buy_etf_when_cheap", "test_strategy_is_net_profitable_after_realistic_costs"],
                          ["test_zscore_serie_constante", "test_signal_bornes_inclusives"]),
    "mm5_e1": Suite(B, "mm_5.py", ["test_tickers_are_clean", "test_numeric_dtypes"], ["test_row_count", "test_no_bad_volumes_survive", "test_no_duplicates_survive"]),
    "mm5_e2": Suite(B, "mm_5.py", ["test_realized_vol_matches_expected"], []),
    "recall_p1": SuiteDouble(R, "recall.py", ["test_clean_prices"],
                             ["test_clean_prices_does_not_mutate_the_caller", "test_clean_prices_treats_existing_nans_like_bad_prints",
                              "test_clean_prices_keeps_the_first_row_of_a_repeated_timestamp", "test_clean_prices_accepts_string_timestamps"],
                             ["test_clean_prices_tri_stable_sur_un_grand_releve", "test_clean_prices_horodatages_convertis"]),
    "recall_p2": Suite(R, "recall.py", ["test_liquidity_scores_in_range_and_sane_ordering"],
                       ["test_liquidity_scores_zero_range_defaults_to_half", "test_liquidity_scores_ignores_extra_request_columns", "test_liquidity_scores_are_rounded_to_six_decimals"]),
    "recall_p3": Suite(R, "recall.py", ["test_reference_price_exact_match", "test_reference_price_skips_nulls_and_within_staleness"],
                       ["test_reference_price_stale_returns_none", "test_reference_price_before_first_print_returns_none", "test_reference_price_exactly_at_staleness_boundary_is_ok",
                        "test_reference_price_uses_most_recent_valid_not_highest_value", "test_reference_price_skips_non_positive_values",
                        "test_reference_price_returns_none_if_no_valid_price_at_or_before_ts", "test_reference_price_staleness_counts_whole_days",
                        "test_reference_price_ignores_other_assets_validity", "test_reference_price_does_not_mutate_the_caller", "test_reference_price_accepts_a_timestamp_object"]),
    "recall_p4": Suite(R, "recall.py", ["test_quote_request_buy_and_sell"], ["test_quote_request_lands_exactly_on_tick_grid", "test_quote_request_bigger_size_worsens_price", "test_quote_request_sell_lands_exactly_on_tick_grid"]),
    "recall_p5": Suite(R, "recall.py", ["test_quote_impact_stream_matches_expected_sequence"],
                       ["test_quote_impact_stream_full_liquidity_means_no_impact", "test_quote_impact_stream_empty_requests", "test_quote_impact_stream_first_quote_excludes_its_own_impact", "test_quote_impact_stream_decay_of_one_keeps_the_whole_state"]),
    "recall_p6": Suite(R, "recall.py", ["test_track_book"], ["test_track_book_empty_stays_flat", "test_track_book_untraded_asset_stays_flat"]),
    "desk_p1": Suite(D, "desk.py", ["test_reconcile_fills"], ["test_reconcile_fills_keeps_the_earliest_of_a_repeated_trade_id", "test_reconcile_fills_all_rows_invalid_returns_an_empty_frame"]),
    "desk_p2": Suite(D, "desk.py", ["test_execution_quality_scores"], ["test_execution_quality_scores_zero_range_defaults_to_half", "test_execution_quality_scores_uses_the_mean_not_the_total"]),
    "desk_p3": Suite(D, "desk.py", ["test_batch_reference_prices"], ["test_batch_reference_prices_empty_queries", "test_batch_reference_prices_staleness_boundary_is_inclusive"]),
    "desk_p4": Suite(D, "desk.py", ["test_enrich_fills_uses_the_latest_reference_row"], ["test_enrich_fills_never_multiplies_rows", "test_enrich_fills_flags_unknown_assets_instead_of_dropping_them", "test_enrich_fills_empty_input_still_has_the_columns"]),
    "desk_p5": Suite(D, "desk.py", ["test_bar_summary_buckets_and_weights"], ["test_bar_summary_share_is_the_bars_total_not_the_whole_tape", "test_bar_summary_prev_vwap_does_not_leak_between_assets", "test_bar_summary_leaves_empty_bars_out"]),
    "desk_p6": Suite(D, "desk.py", ["test_track_book_with_realized"], ["test_track_book_with_realized_reconciles", "test_track_book_with_realized_untraded_asset_stays_flat"]),
    "desk_p7": Suite(D, "desk.py", ["test_daily_scorecard_stats_in_one_aggregation"],
                     ["test_daily_scorecard_single_fill_has_no_price_std", "test_daily_scorecard_pivots_to_one_column_per_asset", "test_daily_scorecard_untraded_cell_is_zero_not_nan",
                      "test_daily_scorecard_cumulative_accumulates_down_the_dates", "test_daily_scorecard_empty_input"]),
    "numpy_p1": Suite(N, "vectorisation.py", ["test_rolling_zscore_basic"], ["test_rolling_zscore_uses_sample_std_not_population", "test_rolling_zscore_flat_window_is_zero_not_inf", "test_rolling_zscore_window_longer_than_series", "test_rolling_zscore_window_of_one"]),
    "numpy_p2": Suite(N, "vectorisation.py", ["test_asof_indices_backward_and_exact_match"], ["test_asof_indices_before_first_print_is_minus_one", "test_asof_indices_staleness_boundary_is_inclusive", "test_asof_indices_unsorted_queries_and_empty"]),
    "numpy_p3": Suite(N, "vectorisation.py", ["test_vwap_by_label_weights_by_size_and_sorts_labels"], ["test_vwap_by_label_handles_unsorted_repeated_labels", "test_vwap_by_label_zero_size_is_nan", "test_vwap_by_label_empty_input"]),
    "numpy_p4": Suite(N, "vectorisation.py", ["test_max_drawdown_basic"], ["test_max_drawdown_monotonic_curve", "test_max_drawdown_reports_first_of_tied_troughs", "test_max_drawdown_empty"]),
    "numpy_p5": Suite(N, "vectorisation.py", ["test_factor_betas_recovers_exact_relationship"], ["test_factor_betas_are_multivariate_not_univariate", "test_factor_betas_constant_asset_returns_has_zero_r2", "test_factor_betas_survives_rank_deficient_factors"]),
    "numpy_p6": Suite(N, "vectorisation.py", ["test_top_k_indices_orders_descending"], ["test_top_k_indices_breaks_ties_toward_lower_column", "test_top_k_indices_ranks_nan_last", "test_top_k_indices_degenerate_k"]),
    "numpy_p7": Suite(N, "vectorisation.py", ["test_winsorize_zscore_clips_before_standardizing"], ["test_winsorize_zscore_keeps_nans_and_ignores_them", "test_winsorize_zscore_flat_after_clipping", "test_winsorize_zscore_all_nan"]),
    "numpy_p8": Suite(N, "vectorisation.py", ["test_ffill_2d_fills_down_each_column_independently"], ["test_ffill_2d_never_backfills", "test_ffill_2d_does_not_mutate_input", "test_ffill_2d_all_nan_column"]),
    "numpy_p9": Suite(N, "vectorisation.py", ["test_book_exposure_applies_the_fx_rate_per_asset"], ["test_book_exposure_weights_sum_to_one_per_day", "test_book_exposure_hedged_day_has_zero_weights", "test_book_exposure_does_not_tile_the_fx_rates"]),
}
