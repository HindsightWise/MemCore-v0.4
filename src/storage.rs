use std::sync::Arc;
use tokio::runtime::Runtime;
use dirs::home_dir;
use crate::models::MemoryEntry;

use lancedb::connect;
use kuzu::{Database, Connection as KuzuConnection, SystemConfig};

use arrow_array::{RecordBatch, StringArray, Float32Array, FixedSizeListArray, Array, RecordBatchIterator};
use arrow_schema::{Schema, Field, DataType};

pub struct StorageController {
    rt: Runtime,
    db_dir: String,
}

impl StorageController {
    pub fn new() -> Self {
        let rt = Runtime::new().expect("Failed to start Tokio runtime for storage");
        let home = home_dir().expect("Unable to find HOME directory");
        let data_dir = home.join(".hermes").join("memcore_data");
        
        std::fs::create_dir_all(&data_dir).expect("Failed to create MemCore database directory");
        
        Self {
            rt,
            db_dir: data_dir.to_string_lossy().to_string(),
        }
    }

    pub fn persist(&self, entry: &MemoryEntry) -> Result<(), String> {
        let lance_path = format!("{}/vectors.lance", self.db_dir);
        let kuzu_path = format!("{}/kuzu_graph", self.db_dir);
        
        // 1. KùzuDB Graph Operation (Zero-Copy)
        // We open the database connection synchronously for the property graph.
        {
            let db = Database::new(&kuzu_path, SystemConfig::default())
                .map_err(|e| format!("Kuzu DB open failed: {:?}", e))?;
            let conn = KuzuConnection::new(&db)
                .map_err(|e| format!("Kuzu connection failed: {:?}", e))?;
            
            // Try to create the Node schema if it doesn't exist
            let _ = conn.query("CREATE NODE TABLE Memory (id STRING, agent_id STRING, text STRING, PRIMARY KEY (id))");
            
            // Sanitize query manually (in production we bind parameters properly)
            let safe_text = entry.text.replace("'", "''");
            let safe_agent = entry.agent_id.replace("'", "''");
            
            let query = format!(
                "CREATE (m:Memory {{id: '{}', agent_id: '{}', text: '{}'}})", 
                entry.id, safe_agent, safe_text
            );
            
            conn.query(&query).map_err(|e| format!("Kuzu Write failed: {:?}", e))?;
        }

        // 2. LanceDB Vector Storage
        // We rely on the async backend via our Tokio runtime.
        if let Some(emb) = &entry.embedding {
            let dim_len = emb.len() as i32;
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("text", DataType::Utf8, false),
                Field::new("agent_id", DataType::Utf8, false),
                Field::new("vector", DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim_len
                ), true),
            ]));

            let id_arr = Arc::new(StringArray::from(vec![entry.id.to_string()])) as Arc<dyn Array>;
            let text_arr = Arc::new(StringArray::from(vec![entry.text.clone()])) as Arc<dyn Array>;
            let agent_arr = Arc::new(StringArray::from(vec![entry.agent_id.clone()])) as Arc<dyn Array>;
            
            let floats = Arc::new(Float32Array::from(emb.clone()));
            let list_builder = Arc::new(FixedSizeListArray::new(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim_len,
                floats,
                None
            )) as Arc<dyn Array>;

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![id_arr, text_arr, agent_arr, list_builder],
            ).map_err(|e| format!("Arrow batch creation error: {:?}", e))?;

            self.rt.block_on(async {
                let connection = connect(&lance_path).execute().await
                    .map_err(|e| format!("LanceDB connection error: {}", e))?;
                
                let tables = connection.table_names().execute().await
                    .map_err(|e| format!("LanceDB list error: {}", e))?;

                if tables.contains(&"memories".to_string()) {
                    let table = connection.open_table("memories").execute().await
                        .map_err(|e| format!("LanceDB open error: {}", e))?;
                    // Add new batch using standard Arrow 50 RecordBatchIterator
                    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
                    table.add(Box::new(batches)).execute().await
                        .map_err(|e| format!("LanceDB write error: {}", e))?;
                } else {
                    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
                    connection.create_table("memories", Box::new(batches)).execute().await
                        .map_err(|e| format!("LanceDB create error: {}", e))?;
                }
                
                Ok::<(), String>(())
            })?;
        }
        
        Ok(())
    }
}
