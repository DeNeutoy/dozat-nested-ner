from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, AdjacencyField, MetadataField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer
from allennlp.data.dataset_readers.dataset_utils.span_utils import (
    bio_tags_to_spans,
    TypedStringSpan,
)


def four_line_chunks(open_file):

    lines = []
    for line in open_file:
        if len(lines) == 4:
            yield lines
            lines = []
        lines.append(line)


@DatasetReader.register("genia_v3")
class GeniaV3NestedNerReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterable[Instance]:

        with open(file_path, "r") as lines:

            for chunk in four_line_chunks(lines):
                words, pos, entities, _ = chunk
                words = words.strip()

                labeled_spans = []
                if entities.strip():
                    entities = entities.strip().split("|")
                    span_set = set()
                    for ent in entities:
                        span, label = ent.split(" ")
                        start, end = span.split(",")
                        label = label.strip("G#")
                        inclusive_span = (int(start), int(end) - 1)
                        if not inclusive_span in span_set:
                            labeled_spans.append((label, inclusive_span))
                            span_set.add(inclusive_span)

                yield self.text_to_instance(words, labeled_spans)

    def text_to_instance(
        self, text: str, labeled_spans: List[TypedStringSpan] = None
    ) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        text_field = TextField(tokens, self.token_indexers)

        fields = {
            "tokens": text_field,
        }
        meta = {"tokens": [t.text for t in tokens]}

        if labeled_spans is not None:
            if not labeled_spans:
                labels = []
                spans = []
            else:
                labels, spans = zip(*labeled_spans)
            adjacency = AdjacencyField(spans, text_field, labels)
            meta["gold"] = labeled_spans
            fields["span_labels"] = adjacency

        fields["metadata"] = MetadataField(meta)

        return Instance(fields)
