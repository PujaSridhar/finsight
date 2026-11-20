from finsight.metrics import aggregate, extract_final_answer, parse_number, score


def test_extract_final_answer_takes_last_answer_line():
    text = "Step 1: 5829 - 5735 = 94\nAnswer: 94\nwait\nAnswer: 95"
    assert extract_final_answer(text) == ("95", True)
    assert extract_final_answer("just 42") == ("just 42", False)


def test_parse_number_formats():
    assert parse_number("$1,577.00") == 1577.0
    assert parse_number("-3.2%") == -3.2
    assert parse_number("(32)") == -32.0
    assert parse_number("about 14.1 percent") == 14.1
    assert parse_number("no numbers") is None


def test_percent_and_decimal_forms_match():
    assert score("Answer: 14.1%", "0.141")["correct"]
    assert score("Answer: 0.141", "14.1%")["correct"]
    assert score("Answer: $1,577 million", "$1577.00")["correct"]


def test_numeric_tolerance_is_one_percent():
    assert score("Answer: 99.5", "100")["correct"]
    assert not score("Answer: 97", "100")["correct"]


def test_yes_no():
    assert score("Answer: Yes, it grew.", "yes")["correct"]
    assert not score("Answer: No", "yes")["correct"]


def test_span_answers():
    assert score("Answer: Rate of inflation; Discount rate", "Rate of inflation; Discount rate")["correct"]


def test_aggregate():
    s = [score("Answer: 1", "1"), score("Answer: 2", "1"), score("no line", "1")]
    agg = aggregate(s)
    assert agg["n"] == 3 and agg["accuracy"] == round(1 / 3, 4) and agg["format_ok"] == round(2 / 3, 4)
