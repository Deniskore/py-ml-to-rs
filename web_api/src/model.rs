use std::sync::Arc;

use ahash::AHashMap;
use anyhow::{Context, Result};
use std::sync::OnceLock;
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

const MAX_DATA_LENGTH: usize = 128;

static CHARS_TO_INDEX_MAP: OnceLock<AHashMap<char, f32>> = OnceLock::new();

pub struct TModel {
    bundle: SavedModelBundle,
    input_op: tensorflow::Operation,
    output_op: tensorflow::Operation,
}

pub fn load_model(
    model_dir: String,
    input_param_name: &str,
    output_param_name: &str,
) -> Result<TModel> {
    let mut graph = Graph::new();
    let bundle = SavedModelBundle::load(&SessionOptions::new(), ["serve"], &mut graph, model_dir)?;

    // Get signature metadata from the model bundle
    let signature = bundle.meta_graph_def().get_signature("serving_default")?;

    // Get input/output info
    let input_info = signature.get_input(input_param_name)?;
    let output_info = signature.get_output(output_param_name)?;

    // Get input/output from graph
    let input_op = graph.operation_by_name_required(&input_info.name().name)?;
    let output_op = graph.operation_by_name_required(&output_info.name().name)?;

    Ok(TModel {
        bundle,
        input_op,
        output_op,
    })
}

fn preprocess_string(input: &str) -> Vec<f32> {
    CHARS_TO_INDEX_MAP.get_or_init(|| {
        // preprocessing map like in Python
        let chars: Vec<char> = (0..128).map(char::from).collect();
        let char_to_index: AHashMap<char, f32> = chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, (i + 1) as f32))
            .collect();

        char_to_index
    });

    let mut result: Vec<f32> = Vec::new();

    if let Some(chars_to_index_map) = CHARS_TO_INDEX_MAP.get() {
        for c in input.chars() {
            if let Some(&index) = chars_to_index_map.get(&c) {
                result.push(index);
            }
        }
    }

    result.resize(MAX_DATA_LENGTH, 0.0);

    result
}

pub fn predict(model: &Arc<TModel>, data: String) -> Result<[f32; 6]> {
    let tensor: Tensor<f32> = Tensor::new(&[1, 128])
        .with_values(&preprocess_string(&data))
        .context("Failed to create tensor")?;

    let mut session_args = SessionRunArgs::new();

    session_args.add_feed(&model.input_op, 0, &tensor);

    let out = session_args.request_fetch(&model.output_op, 0);
    model.bundle.session.run(&mut session_args)?;

    // Fetch outputs
    let out_res_slice: &[f32] = &session_args.fetch(out)?[0..6];

    Ok(out_res_slice.try_into()?)
}
