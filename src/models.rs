use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::HashSet;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MemoryEntry {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub agent_id: String,
    pub text: String,
    pub embedding: Option<Vec<f32>>,         // 384-dimensional from sentence-transformers
    pub kg_node_id: Option<String>,          // KùzuDB Zero-Copy link
    pub skill_tuple: SkillTuple,             // S = {T, O, C}
    pub metadata: serde_json::Value,
    pub version: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SkillTuple {
    pub t: TransformStrategy,
    pub o: HashSet<OperatorMask>,
    pub c: Vec<CompiledConstraint>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TransformStrategy { NormalizeSchema, EuclideanProjection, PassThrough }

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OperatorMask { InsertNode, UpdateEdge, DeleteSubGraph, ReadOnly }

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum CompiledConstraint {
    NoOscillation { window_size: u8 },       // Prevents Checkerboarding
    MaxBudgetSat { percent: f32 },           // Finite memory constraints
    GraphAcyclicity,                         // Drucker-style logical stability
    EvalScript(String),                      // Rhai AST for soft agent rules
}
