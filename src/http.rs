use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;

use crate::Engine;
use crate::continuity::{UciRequest, UciResponse, UnifiedContinuityInterface};
use crate::model::{
    ContextPack, ContextPackManifest, DimensionValue, EventInput, HandoffInput, HandoffRecord,
    IngestManifest, MemoryLayer, MemoryRecord, MetricsSnapshot, QueryInput, RelationRecord,
    ReplayRow, Selector, SubscriptionInput, SubscriptionPoll, SubscriptionRecord, ViewInput,
    ViewManifest, ViewRecord,
};

#[derive(Debug, Deserialize)]
pub struct ReplayParams {
    pub session: Option<String>,
    pub limit: Option<usize>,
    pub selector_json: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryParams {
    pub layer: Option<MemoryLayer>,
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct PollParams {
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct AnnotateRequest {
    pub item_type: String,
    pub item_id: String,
    pub dimensions: Vec<DimensionValue>,
}

#[derive(Debug, Deserialize)]
pub struct RelateRequest {
    pub source_id: String,
    pub target_id: String,
    pub relation: String,
    pub weight: Option<f64>,
    pub attributes: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct ForkRequest {
    pub owner_agent_id: Option<String>,
}

pub async fn serve(engine: Arc<Engine>, addr: SocketAddr) -> Result<()> {
    let app = Router::new()
        .route("/v1/events", post(ingest))
        .route("/v1/context-pack", post(context_pack))
        .route("/v1/context-pack/{id}/explain", get(explain_context_pack))
        .route("/v1/replay", get(replay))
        .route("/v1/memory", get(memory))
        .route("/v1/annotate", post(annotate))
        .route("/v1/relate", post(relate))
        .route("/v1/views", post(create_view))
        .route("/v1/views/{id}", get(get_view))
        .route("/v1/views/{id}/explain", get(explain_view))
        .route("/v1/views/{id}/fork", post(fork_view))
        .route("/v1/handoffs", post(create_handoff))
        .route("/v1/handoffs/{id}", get(get_handoff))
        .route("/v1/handoffs/{id}/explain", get(explain_handoff))
        .route("/v1/subscriptions", post(create_subscription))
        .route("/v1/subscriptions/{id}/poll", get(poll_subscription))
        .route("/v1/uci", post(uci))
        .route("/metrics", get(metrics))
        .with_state(engine);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn ingest(
    State(engine): State<Arc<Engine>>,
    Json(payload): Json<EventInput>,
) -> Result<Json<IngestManifest>, HttpError> {
    Ok(Json(engine.ingest(payload).map_err(HttpError::from)?))
}

async fn context_pack(
    State(engine): State<Arc<Engine>>,
    Json(payload): Json<QueryInput>,
) -> Result<Json<ContextPack>, HttpError> {
    Ok(Json(
        engine
            .build_context_pack(payload)
            .map_err(HttpError::from)?,
    ))
}

async fn explain_context_pack(
    State(engine): State<Arc<Engine>>,
    Path(id): Path<String>,
) -> Result<Json<ContextPackManifest>, HttpError> {
    Ok(Json(
        engine.explain_context_pack(&id).map_err(HttpError::from)?,
    ))
}

async fn replay(
    State(engine): State<Arc<Engine>>,
    Query(params): Query<ReplayParams>,
) -> Result<Json<Vec<ReplayRow>>, HttpError> {
    let rows = if let Some(selector_json) = params.selector_json {
        let selector: Selector = serde_json::from_str(&selector_json)
            .map_err(|err| HttpError(anyhow::Error::new(err)))?;
        engine
            .replay_by_selector(&selector, params.limit.unwrap_or(50))
            .map_err(HttpError::from)?
    } else {
        engine
            .replay(params.session.as_deref(), params.limit.unwrap_or(50))
            .map_err(HttpError::from)?
    };
    Ok(Json(rows))
}

async fn memory(
    State(engine): State<Arc<Engine>>,
    Query(params): Query<MemoryParams>,
) -> Result<Json<Vec<MemoryRecord>>, HttpError> {
    Ok(Json(
        engine
            .list_memory(params.layer, params.limit.unwrap_or(50))
            .map_err(HttpError::from)?,
    ))
}

async fn annotate(
    State(engine): State<Arc<Engine>>,
    Json(payload): Json<AnnotateRequest>,
) -> Result<Json<Vec<DimensionValue>>, HttpError> {
    Ok(Json(
        engine
            .annotate_item(&payload.item_type, &payload.item_id, &payload.dimensions)
            .map_err(HttpError::from)?,
    ))
}

async fn relate(
    State(engine): State<Arc<Engine>>,
    Json(payload): Json<RelateRequest>,
) -> Result<Json<RelationRecord>, HttpError> {
    Ok(Json(
        engine
            .relate_items(
                &payload.source_id,
                &payload.target_id,
                &payload.relation,
                payload.weight.unwrap_or(1.0),
                payload.attributes.unwrap_or_else(|| serde_json::json!({})),
            )
            .map_err(HttpError::from)?,
    ))
}

async fn create_view(
    State(engine): State<Arc<Engine>>,
    Json(payload): Json<ViewInput>,
) -> Result<Json<ViewRecord>, HttpError> {
    Ok(Json(
        engine.materialize_view(payload).map_err(HttpError::from)?,
    ))
}

async fn get_view(
    State(engine): State<Arc<Engine>>,
    Path(id): Path<String>,
) -> Result<Json<ViewRecord>, HttpError> {
    Ok(Json(engine.get_view(&id).map_err(HttpError::from)?))
}

async fn explain_view(
    State(engine): State<Arc<Engine>>,
    Path(id): Path<String>,
) -> Result<Json<ViewManifest>, HttpError> {
    Ok(Json(engine.explain_view(&id).map_err(HttpError::from)?))
}

async fn fork_view(
    State(engine): State<Arc<Engine>>,
    Path(id): Path<String>,
    Json(payload): Json<ForkRequest>,
) -> Result<Json<ViewRecord>, HttpError> {
    Ok(Json(
        engine
            .fork_view(&id, payload.owner_agent_id)
            .map_err(HttpError::from)?,
    ))
}

async fn create_handoff(
    State(engine): State<Arc<Engine>>,
    Json(payload): Json<HandoffInput>,
) -> Result<Json<HandoffRecord>, HttpError> {
    Ok(Json(
        engine.create_handoff(payload).map_err(HttpError::from)?,
    ))
}

async fn get_handoff(
    State(engine): State<Arc<Engine>>,
    Path(id): Path<String>,
) -> Result<Json<HandoffRecord>, HttpError> {
    Ok(Json(engine.get_handoff(&id).map_err(HttpError::from)?))
}

async fn explain_handoff(
    State(engine): State<Arc<Engine>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, HttpError> {
    Ok(Json(engine.explain_handoff(&id).map_err(HttpError::from)?))
}

async fn create_subscription(
    State(engine): State<Arc<Engine>>,
    Json(payload): Json<SubscriptionInput>,
) -> Result<Json<SubscriptionRecord>, HttpError> {
    Ok(Json(
        engine
            .create_subscription(payload)
            .map_err(HttpError::from)?,
    ))
}

async fn poll_subscription(
    State(engine): State<Arc<Engine>>,
    Path(id): Path<String>,
    Query(params): Query<PollParams>,
) -> Result<Json<SubscriptionPoll>, HttpError> {
    Ok(Json(
        engine
            .poll_subscription(&id, params.limit.unwrap_or(25))
            .map_err(HttpError::from)?,
    ))
}

async fn uci(
    State(engine): State<Arc<Engine>>,
    Json(payload): Json<UciRequest>,
) -> Result<Json<UciResponse>, HttpError> {
    Ok(Json(
        engine.handle_request(payload).map_err(HttpError::from)?,
    ))
}

async fn metrics(State(engine): State<Arc<Engine>>) -> Result<String, HttpError> {
    let snapshot: MetricsSnapshot = engine.metrics_snapshot().map_err(HttpError::from)?;
    Ok(snapshot.prometheus_text)
}

#[derive(Debug)]
struct HttpError(anyhow::Error);

impl From<anyhow::Error> for HttpError {
    fn from(value: anyhow::Error) -> Self {
        Self(value)
    }
}

impl IntoResponse for HttpError {
    fn into_response(self) -> axum::response::Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": self.0.to_string() })),
        )
            .into_response()
    }
}
