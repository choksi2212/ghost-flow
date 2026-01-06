//! GhostFlow Model Serving REST API
//!
//! A production-ready REST API server for serving GhostFlow models.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tower_http::cors::CorsLayer;
use tracing::{info, warn};
use uuid::Uuid;

mod handlers;
mod models;

use models::{ModelRegistry, PredictionRequest, PredictionResponse};

/// Application state
#[derive(Clone)]
struct AppState {
    registry: Arc<RwLock<ModelRegistry>>,
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("ghostflow_serve=info,tower_http=debug")
        .init();

    info!("Starting GhostFlow Model Serving API");

    // Create application state
    let state = AppState {
        registry: Arc::new(RwLock::new(ModelRegistry::new())),
    };

    // Build router
    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health_check))
        .route("/models", get(list_models))
        .route("/models/:id", get(get_model))
        .route("/models/:id/predict", post(predict))
        .route("/models/load", post(load_model))
        .route("/models/:id/unload", post(unload_model))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Start server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
        .await
        .unwrap();
    
    info!("Server listening on http://0.0.0.0:8080");
    
    axum::serve(listener, app).await.unwrap();
}

/// Root endpoint
async fn root() -> &'static str {
    "GhostFlow Model Serving API v0.1.0"
}

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))
}

/// List all loaded models
async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let registry = state.registry.read().unwrap();
    let models = registry.list_models();
    Json(models)
}

/// Get model information
async fn get_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    let registry = state.registry.read().unwrap();
    
    match registry.get_model(&id) {
        Some(model) => Ok(Json(model)),
        None => Err(StatusCode::NOT_FOUND),
    }
}

/// Make a prediction
async fn predict(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<PredictionRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    info!("Prediction request for model: {}", id);
    
    let registry = state.registry.read().unwrap();
    
    match registry.predict(&id, request) {
        Ok(response) => Ok(Json(response)),
        Err(e) => {
            warn!("Prediction failed: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Load a model
async fn load_model(
    State(state): State<AppState>,
    Json(request): Json<LoadModelRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    info!("Loading model: {}", request.name);
    
    let mut registry = state.registry.write().unwrap();
    
    match registry.load_model(request.name, request.path) {
        Ok(id) => Ok(Json(serde_json::json!({
            "id": id,
            "status": "loaded",
        }))),
        Err(e) => {
            warn!("Failed to load model: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Unload a model
async fn unload_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    info!("Unloading model: {}", id);
    
    let mut registry = state.registry.write().unwrap();
    
    match registry.unload_model(&id) {
        Ok(_) => Ok(Json(serde_json::json!({
            "status": "unloaded",
        }))),
        Err(e) => {
            warn!("Failed to unload model: {}", e);
            Err(StatusCode::NOT_FOUND)
        }
    }
}

#[derive(Debug, Deserialize)]
struct LoadModelRequest {
    name: String,
    path: String,
}
