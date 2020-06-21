
from nested_ner.genia_reader import GeniaNestedNerReader


class TestNestedNerReader:
    def test_read_from_file(self):
        reader = GeniaNestedNerReader()
        data_path = "tests/fixtures/data.iob2"
        instances = reader.read(data_path)

        assert len(instances) == 2

        fields = instances[0].fields
        expected_tokens = ["it", "is", "movies", "like", "these"]
        assert [t.text for t in fields["text"].tokens][:5] == expected_tokens
        assert fields["label"].label == "neg"

        fields = instances[1].fields
        expected_tokens = ["the", "music", "is", "well-chosen", "and"]
        assert [t.text for t in fields["text"].tokens][:5] == expected_tokens
        assert fields["label"].label == "pos"
