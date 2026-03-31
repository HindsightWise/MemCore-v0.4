use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use uuid::Uuid;
use chrono::Utc;

mod models; mod dnc; mod legislator; mod storage;
use models::{MemoryEntry, SkillTuple};
use dnc::{DncController, DncObservation};
use legislator::SemanticLegislator;
use storage::StorageController;

#[pyclass]
pub struct MemCoreEngine {
    dnc: DncController,
    legislator: SemanticLegislator,
    storage: StorageController,
}

#[pymethods]
impl MemCoreEngine {
    #[new]
    fn new() -> Self {
        Self { 
            dnc: DncController::new(), 
            legislator: SemanticLegislator::new(),
            storage: StorageController::new(),
        }
    }

    fn store(
        &mut self, agent_id: String, text: String, skill_tuple_json: String, 
        dnc_obs_json: String, temperature: f32, embedding_json: Option<String>
    ) -> PyResult<String> {
        let skill_tuple: SkillTuple = serde_json::from_str(&skill_tuple_json).unwrap();
        let obs: DncObservation = serde_json::from_str(&dnc_obs_json).unwrap();
        
        let embedding: Option<Vec<f32>> = match embedding_json {
            Some(j) => serde_json::from_str(&j).unwrap_or(None),
            None => None
        };

        // 1. Direct Numeric Control (Grayness Gate)
        if let Err(e) = self.dnc.evaluate(&obs, temperature) {
            return Err(PyValueError::new_err(format!("[DNC REJECTION] {}", e)));
        }

        let entry = MemoryEntry {
            id: Uuid::new_v4(), timestamp: Utc::now(), agent_id, text,
            embedding, kg_node_id: None, skill_tuple,
            metadata: serde_json::json!({}), version: 1,
        };

        // 2. Layer 1 Semantic Legislator S={T,O,C} Validation
        if let Err(e) = self.legislator.validate(&entry) {
            return Err(PyValueError::new_err(format!("[CONSTRAINT VIOLATION] {}", e)));
        }

        // 3. (Production) Execute hybrid Kùzu/LanceDB write
        if let Err(e) = self.storage.persist(&entry) {
             return Err(PyValueError::new_err(format!("[STORAGE REJECTION] {}", e)));
        }
        Ok(format!("Memory {} validated and committed.", entry.id))
    }
}

#[pymodule]
fn memcore(_py: Python, m: &PyModule) -> PyResult<()> { m.add_class::<MemCoreEngine>()?; Ok(()) }
