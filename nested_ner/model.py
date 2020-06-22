from typing import Dict, Tuple, Any, List
import logging
import copy

from overrides import overrides
import torch
from torch.nn.modules import Dropout
import numpy

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, Activation
from allennlp.nn.util import min_value_of_dtype
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.dataset_readers.dataset_utils.span_utils import TypedStringSpan

from nested_ner.per_class_f1 import PerClassScorer

logger = logging.getLogger(__name__)


@Model.register("dozat_nested_ner")
class DozatNestedNer(Model):
    """
    A Nested NER Model.
    Registered as a `Model` with name "dozat_nested_ner".
    # Parameters
    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : `TextFieldEmbedder`, required
        Used to embed the `tokens` `TextField` we get as input to the model.
    encoder : `Seq2SeqEncoder`
        The encoder (with its own internal stacking) that we will use to generate representations
        of tokens.
    tag_representation_dim : `int`, required.
        The dimension of the MLPs used for span tag prediction.
    span_representation_dim : `int`, required.
        The dimension of the MLPs used for span prediction.
    tag_feedforward : `FeedForward`, optional, (default = None).
        The feedforward network used to produce tag representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    span_feedforward : `FeedForward`, optional, (default = None).
        The feedforward network used to produce span representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    dropout : `float`, optional, (default = 0.0)
        The variational dropout applied to the output of the encoder and MLP layers.
    input_dropout : `float`, optional, (default = 0.0)
        The dropout applied to the embedded text input.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        tag_representation_dim: int,
        span_representation_dim: int,
        tag_feedforward: FeedForward = None,
        span_feedforward: FeedForward = None,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        span_prediction_threshold: float = 0.5,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.span_prediction_threshold = span_prediction_threshold
        encoder_dim = encoder.get_output_dim()

        self.head_span_feedforward = span_feedforward or FeedForward(
            encoder_dim, 1, span_representation_dim, Activation.by_name("elu")()
        )
        self.child_span_feedforward = copy.deepcopy(self.head_span_feedforward)

        self.span_attention = BilinearMatrixAttention(
            span_representation_dim, span_representation_dim, use_input_biases=True
        )

        num_labels = self.vocab.get_vocab_size("labels")
        self.head_tag_feedforward = tag_feedforward or FeedForward(
            encoder_dim, 1, tag_representation_dim, Activation.by_name("elu")()
        )
        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = BilinearMatrixAttention(
            tag_representation_dim, tag_representation_dim, label_dim=num_labels
        )

        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)

        representation_dim = text_field_embedder.get_output_dim()

        check_dimensions_match(
            representation_dim,
            encoder.get_input_dim(),
            "text field embedding dim",
            "encoder input dim",
        )
        check_dimensions_match(
            tag_representation_dim,
            self.head_tag_feedforward.get_output_dim(),
            "tag representation dim",
            "tag feedforward output dim",
        )
        check_dimensions_match(
            span_representation_dim,
            self.head_span_feedforward.get_output_dim(),
            "span representation dim",
            "span feedforward output dim",
        )

        self._span_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self._tag_loss = torch.nn.CrossEntropyLoss(reduction="none")

        self._scorer = PerClassScorer()
        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        metadata: List[Dict[str, Any]] = None,
        span_labels: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters
        tokens : TextFieldTensors, required
            The output of `TextField.as_array()`.
        metadata : List[Dict[str, Any]], optional (default = None)
            A dictionary of metadata for each batch element which has keys:
                tokens : `List[str]`, required.
                    The original string tokens in the sentence.
        span_labels : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape `(batch_size, sequence_length, sequence_length)`.
        # Returns
        An output dictionary.
        """
        embedded_text_input = self.text_field_embedder(tokens)

        mask = get_text_field_mask(tokens)
        embedded_text_input = self._input_dropout(embedded_text_input)
        encoded_text = self.encoder(embedded_text_input, mask)

        encoded_text = self._dropout(encoded_text)

        # shape (batch_size, sequence_length, span_representation_dim)
        head_span_representation = self._dropout(self.head_span_feedforward(encoded_text))
        child_span_representation = self._dropout(self.child_span_feedforward(encoded_text))

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self._dropout(self.head_tag_feedforward(encoded_text))
        child_tag_representation = self._dropout(self.child_tag_feedforward(encoded_text))
        # shape (batch_size, sequence_length, sequence_length)
        span_scores = self.span_attention(head_span_representation, child_span_representation)
        # shape (batch_size, num_tags, sequence_length, sequence_length)
        span_tag_logits = self.tag_bilinear(head_tag_representation, child_tag_representation)
        # Switch to (batch_size, sequence_length, sequence_length, num_tags)
        span_tag_logits = span_tag_logits.permute(0, 2, 3, 1).contiguous()

        # Since we'll be doing some additions, using the min value will cause underflow
        minus_mask = ~mask * min_value_of_dtype(span_scores.dtype) / 10
        span_scores = span_scores + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        span_probs, span_tag_probs = self._greedy_decode(span_scores, span_tag_logits, mask)

        output_dict = {"span_probs": span_probs, "span_tag_probs": span_tag_probs, "mask": mask}

        if metadata:
            output_dict["tokens"] = [meta["tokens"] for meta in metadata]

        if span_labels is not None:
            span_nll, tag_nll = self._construct_loss(
                span_scores=span_scores, span_tag_logits=span_tag_logits, span_tags=span_labels, mask=mask
            )
            output_dict["loss"] = span_nll + tag_nll
            output_dict["span_loss"] = span_nll
            output_dict["tag_loss"] = tag_nll

            if not self.training:
                batch_predicted = self.span_inference(span_probs.cpu().detach().numpy(), span_tag_probs.cpu().detach().numpy(), mask)
                # Flatten lists of spans
                for predicted, meta in zip(batch_predicted, metadata):
                    self._scorer(predicted, meta["gold"])
        return output_dict


    def span_inference(
        self,
        span_probs: torch.Tensor,
        span_tag_probs: torch.Tensor,
        mask: torch.Tensor
        ) -> List[List[TypedStringSpan]]:

        lengths = get_lengths_from_binary_sequence_mask(mask)
        batch_spans = []
        for instance_span_probs, instance_span_tag_probs, length in zip(
            span_probs, span_tag_probs, lengths
        ):
            span_matrix = instance_span_probs > self.span_prediction_threshold
            spans = []
            for i in range(length):
                # Strict requirement for span starts to be before span ends,
                # so j index starts from i, not 0.
                for j in range(i, length):
                    if span_matrix[i, j] == 1:
                        tag = instance_span_tag_probs[i, j].argmax(-1)
                        string_tag = self.vocab.get_token_from_index(tag, "labels")
                        spans.append((string_tag, (i, j)))
            batch_spans.append(spans)
        return batch_spans

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        span_tag_probs = output_dict["span_tag_probs"].cpu().detach().numpy()
        span_probs = output_dict["span_probs"].cpu().detach().numpy()
        mask = output_dict["mask"]
        lengths = get_lengths_from_binary_sequence_mask(mask)
        output_dict["spans"] = self.span_inference(span_probs, span_tag_probs, mask)
        return output_dict

    def _construct_loss(
        self,
        span_scores: torch.Tensor,
        span_tag_logits: torch.Tensor,
        span_tags: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the span and tag loss for an adjacency matrix.
        # Parameters
        span_scores : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate a
            binary classification decision for whether an edge is present between two words.
        span_tag_logits : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length, num_tags) used to generate
            a distribution over edge tags for a given edge.
        span_tags : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length).
            The labels for every span.
        mask : `torch.BoolTensor`, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.
        # Returns
        span_nll : `torch.Tensor`, required.
            The negative log likelihood from the span loss.
        tag_nll : `torch.Tensor`, required.
            The negative log likelihood from the span tag loss.
        """
        span_indices = (span_tags != -1).float()
        # Make the span tags not have negative values anywhere
        # (by default, no edge is indicated with -1).
        span_tags = span_tags * span_indices
        span_nll = self._span_loss(span_scores, span_indices) * mask.unsqueeze(1) * mask.unsqueeze(2)
        # We want the mask for the tags to only include the unmasked words
        # and we only care about the loss with respect to the gold spans.
        tag_mask = mask.unsqueeze(1) * mask.unsqueeze(2) * span_indices

        batch_size, sequence_length, _, num_tags = span_tag_logits.size()
        original_shape = [batch_size, sequence_length, sequence_length]
        reshaped_logits = span_tag_logits.view(-1, num_tags)
        reshaped_tags = span_tags.view(-1)
        tag_nll = (
            self._tag_loss(reshaped_logits, reshaped_tags.long()).view(original_shape) * tag_mask
        )

        valid_positions = tag_mask.sum()

        span_nll = span_nll.sum() / valid_positions.float()
        tag_nll = tag_nll.sum() / valid_positions.float()
        return span_nll, tag_nll

    @staticmethod
    def _greedy_decode(
        span_scores: torch.Tensor, span_tag_logits: torch.Tensor, mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the span start and span end predictions by decoding the unlabeled spans
        independently for each word and then again, predicting the head tags of
        these greedily chosen spans independently.
        # Parameters
        span_scores : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) representing the
            liklihood that (i, j) is a span.
        span_tag_logits : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length, num_tags) used to
            generate a distribution over tags for each span.
        mask : `torch.BoolTensor`, required.
            A mask of shape (batch_size, sequence_length).

        # Returns
        span_probs : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length, sequence_length) representing the
            probability of an span being present for this edge.
        span_tag_probs : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length, sequence_length, sequence_length)
            representing the distribution over edge tags for a given edge.
        """
        # Mask the lower triangular part of the scores, because we can't have
        # span starts which start after the span ends.
        seq_length = span_scores.shape[-1]
        inf_lower_tri_mask = torch.triu(span_scores.new_ones(seq_length, seq_length)).log()
        span_scores = span_scores + inf_lower_tri_mask
        # shape (batch_size, sequence_length, sequence_length, num_tags)
        span_tag_logits = span_tag_logits + inf_lower_tri_mask.unsqueeze(0).unsqueeze(-1)
        # Mask padded tokens, because we only want to consider actual word -> word edges.
        minus_mask = ~mask.unsqueeze(2)
        span_scores.masked_fill_(minus_mask, -numpy.inf)
        span_tag_logits.masked_fill_(minus_mask.unsqueeze(-1), -numpy.inf)
        # shape (batch_size, sequence_length, sequence_length)
        span_probs = span_scores.sigmoid()
        # shape (batch_size, sequence_length, sequence_length, num_tags)
        span_tag_probs = torch.nn.functional.softmax(span_tag_logits, dim=-1)
        return span_probs, span_tag_probs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._scorer.get_metric(reset=reset)
