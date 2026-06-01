//! Feedback prompt enhancement with spectral analysis.

/// Enhanced feedback for the meta/feedback agents.
pub struct FeedbackEnhancer;

impl FeedbackEnhancer {
    /// Generate a spectral analysis section for the feedback prompt.
    pub fn enhance(
        target_mode: &str,
        contraction: f64,
        conservation_holds: bool,
        spectral_gap: f64,
        information_gain: f64,
        universality: &str,
    ) -> String {
        let converging = contraction < 1.0;
        format!(
            "## SPECTRAL IMPROVEMENT ANALYSIS (SIA²)\n\
             \n\
             ### Target Mode: {target_mode}\n\
             This is the WEAKEST eigenmode. Focus improvements here.\n\
             \n\
             ### Convergence\n\
             - Banach contraction: {contraction:.4}\n\
             - {}\n\
             - Spectral gap: {spectral_gap:.4}\n\
             \n\
             ### Conservation\n\
             - {}\n\
             \n\
             ### Information Geometry\n\
             - Fisher information gained: {information_gain:.4}\n\
             \n\
             ### Renormalization\n\
             - Universality class: {universality}\n\
             \n\
             ### INSTRUCTIONS\n\
             1. Focus on **{target_mode}**\n\
             2. {}\n\
             3. {}\n",
            if converging { "✓ CONVERGING" } else { "⚠ NOT CONVERGING" },
            if conservation_holds { "✓ All laws hold" } else { "✗ VIOLATED — restore capabilities" },
            if conservation_holds { "Respect conservation laws" } else { "RESTORE lost capabilities FIRST" },
            if converging { "Continue current trajectory" } else { "CHANGE strategy — not converging" },
        )
    }

    /// Format eigenmodes for display.
    pub fn format_modes(modes: &[(String, f64, bool)]) -> String {
        modes
            .iter()
            .enumerate()
            .map(|(i, (name, eigenvalue, weak))| {
                let marker = if *weak { "🔴 WEAK" } else { "🟢 strong" };
                format!("  {}. {} λ={:.4} ({})", i + 1, name, eigenvalue, marker)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhance_converging_conserved() {
        let result = FeedbackEnhancer::enhance("mode_0_reasoning", 0.7, true, 0.3, 0.15, "gaussian");
        assert!(result.contains("CONVERGING"));
        assert!(result.contains("All laws hold"));
        assert!(result.contains("mode_0_reasoning"));
    }

    #[test]
    fn test_enhance_not_converging() {
        let result = FeedbackEnhancer::enhance("mode_1", 1.3, true, 0.1, 0.05, "relevant_operator");
        assert!(result.contains("NOT CONVERGING"));
    }

    #[test]
    fn test_enhance_violated_conservation() {
        let result = FeedbackEnhancer::enhance("mode_2", 0.8, false, 0.2, 0.1, "wilson-fisher");
        assert!(result.contains("VIOLATED"));
        assert!(result.contains("RESTORE"));
    }

    #[test]
    fn test_format_modes() {
        let modes = vec![
            ("mode_0".to_string(), 0.9, false),
            ("mode_1".to_string(), 0.05, true),
        ];
        let formatted = FeedbackEnhancer::format_modes(&modes);
        assert!(formatted.contains("strong"));
        assert!(formatted.contains("WEAK"));
    }

    #[test]
    fn test_format_modes_empty() {
        let modes: Vec<(String, f64, bool)> = vec![];
        let formatted = FeedbackEnhancer::format_modes(&modes);
        assert!(formatted.is_empty());
    }

    #[test]
    fn test_enhance_contains_instructions() {
        let result = FeedbackEnhancer::enhance("mode_0", 0.5, true, 0.2, 0.1, "gaussian");
        assert!(result.contains("INSTRUCTIONS"));
        assert!(result.contains("Focus on"));
    }

    #[test]
    fn test_enhance_shows_spectral_gap() {
        let result = FeedbackEnhancer::enhance("mode_0", 0.5, true, 0.42, 0.1, "gaussian");
        assert!(result.contains("0.42"));
    }

    #[test]
    fn test_enhance_shows_universality() {
        let result = FeedbackEnhancer::enhance("mode_0", 0.5, true, 0.2, 0.1, "asymptotic_freedom");
        assert!(result.contains("asymptotic_freedom"));
    }
}
