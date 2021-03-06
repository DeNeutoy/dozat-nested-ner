// For more info on config files generally, see https://guide.allennlp.org/using-config-files
local max_length = 512;
local transformer_model = "bert-base-uncased";

{
    "dataset_reader" : {
        // This name needs to match the name that you used to register your dataset reader, with
        // the call to `@DatasetReader.register()`.
        "type": "genia",
        // These other parameters exactly match the constructor parameters of your dataset reader class.
        "token_indexers": {
            "tokens": {
		"type": "pretrained_transformer_mismatched",
		"model_name": transformer_model,
		"max_length": max_length
	      },
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
		    "type": "pretrained_transformer_mismatched",
		    "model_name": transformer_model,
		    "max_length": max_length
		}
            }
        },
      "encoder": {
        "type": "pass_through",
	"input_dim": 768,
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
        "batch_size" : 64
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
