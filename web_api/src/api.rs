use ntex::web;
use ntex::web::types::Json;
use ntex::web::HttpRequest;
use ntex::web::HttpResponse;
use serde_derive::Deserialize;
use serde_derive::Serialize;
use std::sync::Arc;

use crate::model;
use crate::model::TModel;

const MINIMUM_DATA_LENGTH: usize = 4;

#[derive(Deserialize)]
pub struct PredictRequest {
    language: String,
    data: String,
}

#[derive(Serialize)]
struct PredictResponse {
    text: String,
    rot13: String,
    caesar: String,
    base85: String,
    base64: String,
    base58: String,
}

pub async fn predict(
    req: Json<PredictRequest>,
    model: web::types::State<Arc<TModel>>,
) -> HttpResponse {
    let predict_request = req.into_inner();

    if predict_request.data.is_empty() || predict_request.data.len() < MINIMUM_DATA_LENGTH {
        return HttpResponse::BadRequest().json(&"Input data length must be at least 4 bytes");
    }

    if predict_request.language.to_lowercase() != "english" {
        return HttpResponse::BadRequest().json(&"The language is not supported");
    }

    let predictions = match model::predict(model.get_ref(), predict_request.data) {
        Ok(p) => p,
        Err(err) => return HttpResponse::BadRequest().json(&err.to_string()),
    };

    let predict_response = PredictResponse {
        text: format!("{:.2}", predictions[0] * 100.0),
        rot13: format!("{:.2}", predictions[1] * 100.0),
        caesar: format!("{:.2}", predictions[2] * 100.0),
        base85: format!("{:.2}", predictions[3] * 100.0),
        base64: format!("{:.2}", predictions[4] * 100.0),
        base58: format!("{:.2}", predictions[5] * 100.0),
    };

    HttpResponse::Ok().json(&predict_response)
}

pub async fn index(_req: HttpRequest) -> &'static str {
    "Use POST request to /predict"
}
