from sense_lang.providers import MockModelProvider


def test_mock_provider_default_template():
    provider = MockModelProvider()
    response = provider.complete("hello")
    assert response.text == "(mock reasoning about: hello)"
    assert response.confidence == 0.5


def test_mock_provider_custom_template_and_confidence():
    provider = MockModelProvider(response_template="echo: {prompt}", confidence=0.9)
    response = provider.complete("world")
    assert response.text == "echo: world"
    assert response.confidence == 0.9
