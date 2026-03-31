use crate::models::{CompiledConstraint, MemoryEntry};
use rhai::{Engine, Scope};

pub struct SemanticLegislator {
    rhai_engine: Engine,
}

impl SemanticLegislator {
    pub fn new() -> Self { Self { rhai_engine: Engine::new() } }

    pub fn validate(&self, entry: &MemoryEntry) -> Result<(), String> {
        for constraint in &entry.skill_tuple.c {
            match constraint {
                CompiledConstraint::EvalScript(script) => {
                    let mut scope = Scope::new();
                    scope.push("text_len", entry.text.len() as i64);
                    scope.push("text", entry.text.clone());
                    
                    let result: bool = self.rhai_engine.eval_with_scope(&mut scope, script)
                        .map_err(|e| format!("Rhai execution error: {}", e))?;
                    
                    if !result {
                        return Err(format!("Constraint EvalScript '{}' failed.", script));
                    }
                }
                CompiledConstraint::MaxBudgetSat { percent } => {
                    if *percent > 0.95 { return Err("Memory budget saturation critical (>95%).".into()); }
                }
                _ => {} // Constraints like GraphAcyclicity are passed directly to the DB Layer
            }
        }
        Ok(())
    }
}
