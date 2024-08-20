use std::sync::Arc;

use api::{index, predict};
use ntex::web::{self, App};

const MAX_DATA_LENGTH: usize = 1024;

mod api;
mod model;

#[ntex::main]
async fn main() -> std::io::Result<()> {
    let model = Arc::new(
        model::load_model(
            "../../model/detector".to_string(),
            "predict_input",
            "predict_output",
        )
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?,
    );

    web::server(move || {
        App::new()
            .state(web::types::JsonConfig::default().limit(MAX_DATA_LENGTH))
            .state(model.clone())
            .service((
                web::resource("/predict")
                    .state(web::types::JsonConfig::default().limit(MAX_DATA_LENGTH))
                    .route(web::post().to(predict)),
                web::resource("/").route(web::get().to(index)),
            ))
    })
    .bind("127.0.0.1:3000")?
    .run()
    .await
}
