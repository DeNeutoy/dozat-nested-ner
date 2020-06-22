// For more info on config files generally, see https://guide.allennlp.org/using-config-files
{
    "dataset_reader" : {
        // This name needs to match the name that you used to register your dataset reader, with
        // the call to `@DatasetReader.register()`.
        "type": "genia",
        // These other parameters exactly match the constructor parameters of your dataset reader class.
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        }
    },
    "train_data_path": "/net/nfs.corp/s2-research/markn/data/genia/genia.train.iob2",
    "validation_data_path": "/net/nfs.corp/s2-research/markn/data/genia/genia.dev.iob2",
    "model": {
        // This name needs to match the name that you used to register your dataset reader, with
        // the call to `@DatasetReader.register()`.
        "type": "dozat_nested_ner",
        // These other parameters exactly match the constructor parameters of your model class.
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 100,
                    "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz",
                    "trainable": true
                }
            }
        },
      "encoder": {
        "type": "stacked_bidirectional_lstm",
        "input_size": 100,
        "hidden_size": 200,
        "num_layers": 3,
        "recurrent_dropout_probability": 0.3,
        "use_highway": true
      },
      "span_representation_dim": 500,
      "tag_representation_dim": 100,
      "dropout": 0.3,
      "input_dropout": 0.3,
      "initializer": {
        "regexes": [
          [".*feedforward.*weight", {"type": "xavier_uniform"}],
          [".*feedforward.*bias", {"type": "zero"}],
          [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
          [".*tag_bilinear.*bias", {"type": "zero"}],
          [".*weight_ih.*", {"type": "xavier_uniform"}],
          [".*weight_hh.*", {"type": "orthogonal"}],
          [".*bias_ih.*", {"type": "zero"}],
          [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
        ]
      }
    },
    "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "batch_size" : 256
      }
    },
    "trainer": {
      "num_epochs": 30,
      "grad_norm": 5.0,
      "patience": 20,
      "cuda_device": 0,
      "validation_metric": "+f1-measure-overall",
      "optimizer": {
        "type": "huggingface_adamw",
        "lr": 2e-5,
        "eps": 1e-8
      }
    }
    // There are a few other optional parameters that can go at the top level, e.g., to configure
    // vocabulary behavior, to use a separate dataset reader for validation data, or other things.
    // See http://docs.allennlp.org/master/api/commands/train/#from_partial_objects for more info
    // on acceptable parameters.
}
