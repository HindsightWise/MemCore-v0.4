use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DncObservation {
    pub grayness_index: f32,       // G: Epistemic Uncertainty (e.g., logprob dispersion)
    pub stagnation_counter: u32,
    pub budget_consumption: f32,
    pub relative_change: f32,      // Delta trigger for KGFP radar
    pub checkerboard_measure: f32, // State oscillation measure
}

pub struct DncController {
    max_grayness: f32,
    min_beta: f32,
}

impl DncController {
    pub fn new() -> Self {
        Self { max_grayness: 0.20, min_beta: 8.0 }
    }

    pub fn evaluate(&self, obs: &DncObservation, current_beta: f32) -> Result<(), String> {
        // HARD GRAYNESS GATE: 
        // If agent temperature is low (highly confident) but math dispersion is high, REJECT.
        if current_beta <= self.min_beta && obs.grayness_index > self.max_grayness {
            return Err(format!(
                "HARD GRAYNESS GATE: Epistemic uncertainty (G={:.2}) exceeds safe bounds for current Beta ({:.2}). Silent hallucination risk.",
                obs.grayness_index, current_beta
            ));
        }

        // CHECKERBOARD GATE: Prevent thrashing states (e.g., Yes -> No -> Yes)
        if obs.checkerboard_measure > 0.5 {
            return Err("CHECKERBOARD TRIGGERED: Agent is logically oscillating. Stop thrashing.".into());
        }

        Ok(())
    }
}
