# Intro
A friend of mine asked me to demonstrate how to load a model trained in Python into a Rust service. In response, this repository showcases the entire process of training a machine learning model to distinguish between various text encodings, achieving around 98.5% validation accuracy, using data sourced from the English Wiktionary.
Subsequently, the trained model is seamlessly integrated into a Rust-based microservice, utilizing the ntex-rs. This implementation is streamlined with minimal dependencies, ensuring a lightweight and efficient service.

<br/>
Supported encodings:

1. Plain text
2. Rot13
3. Caesar
4. Base85
5. Base64
6. Base58

## Dependencies

- Python 3.10+
- Rust 1.75+

## Train Model
1. Download the [English Wiktionary dump](https://dumps.wikimedia.org/enwiktionary/).
2. Open `test.ipynb` and modify the variable `wiktionary_dump_filepath` to point to the downloaded dump file.
3. Execute the first cell in the notebook.
4. Execute the second cell in the notebook to train the model (ensure that all dependencies associated with TensorFlow and Keras are properly installed).
5. To evaluate the model, run the third cell.

## Usage of web_api

Run in the web_api directory:
```
cargo run --release
```

Run the following command in terminal:

```
curl -X POST -H "Content-Type: application/json" -d '{"language":"English","data":"HELLO WORLD"}' http://127.0.0.1:3000/predict
```

The prediction will be presented in the following format:

```json
{"text":"99.51","rot13":"0.00","caesar":"0.49","base85":"0.00","base64":"0.00","base58":"0.00"}
```

## License
This project is licensed under the [MIT license](LICENSE).
