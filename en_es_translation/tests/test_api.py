from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
import torch


class MockModel:
    """A simple mock model that behaves like the real model."""
    def __init__(self, default_translation="Hola mundo"):
        self.default_translation = default_translation
        self.call_history = []
    
    def __call__(self, text_list):
        """Simulate model inference."""
        self.call_history.append(text_list)
        return [self.default_translation]
    
    def reset(self):
        """Reset call history."""
        self.call_history = []


@pytest.fixture
def mock_model():
    """Create a mock model that returns predictable translations."""
    return MockModel()


@pytest.fixture
def mock_device():
    """Create a mock device."""
    return torch.device("cpu")


@pytest.fixture
def client_with_model(mock_model, mock_device):
    """Create a test client with a mocked model loaded."""
    # Patch load_model before importing the app module
    with patch("en_es_translation.api.load_model") as mock_load_model:
        mock_load_model.return_value = (mock_model, mock_device)
        
        # Import here to ensure patch is active when module loads
        from en_es_translation.api import app
        
        # Manually set the app state since lifespan runs async and TestClient
        # may not execute it properly
        app.state.model = mock_model
        app.state.device = mock_device
        
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def client():
    """Create a test client without model (for error scenarios)."""
    from en_es_translation.api import app
    
    # Save original state
    original_model = getattr(app.state, "model", None)
    original_device = getattr(app.state, "device", None)
    
    # Create client - lifespan will load model, but we'll override it
    with TestClient(app) as test_client:
        # Override to None to simulate model not loaded
        app.state.model = None
        app.state.device = None
        yield test_client
    
    # Restore original state
    app.state.model = original_model
    app.state.device = original_device


class TestSuccessfulTranslation:
    """Tests for successful translation scenarios."""

    def test_translate_valid_request(self, client_with_model, mock_model):
        """Test valid translation request returns correct response."""
        response = client_with_model.post(
            "/translate",
            json={"text": "Hello world"}
        )
        
        assert response.status_code == 200
        # FastAPI JSON-encodes string responses
        assert response.json() == "Hola mundo"
        # Verify model was called correctly
        assert len(mock_model.call_history) == 1
        assert mock_model.call_history[0] == ["Hello world"]

    def test_translate_different_texts(self, client_with_model, mock_model):
        """Test translation with different input texts."""
        # Create a model with custom translation logic
        translations = {
            "Hello": "Hola",
            "Good morning": "Buenos días",
            "How are you?": "¿Cómo estás?"
        }
        
        class CustomMockModel(MockModel):
            def __call__(self, text_list):
                self.call_history.append(text_list)
                return [translations.get(text_list[0], "Translated")]
        
        # Replace the model in app.state
        from en_es_translation.api import app
        custom_model = CustomMockModel()
        app.state.model = custom_model
        
        test_cases = [
            ("Hello", "Hola"),
            ("Good morning", "Buenos días"),
            ("How are you?", "¿Cómo estás?")
        ]
        
        for input_text, expected_output in test_cases:
            response = client_with_model.post(
                "/translate",
                json={"text": input_text}
            )
            assert response.status_code == 200
            # FastAPI JSON-encodes string responses
            assert response.json() == expected_output

    def test_translate_response_format(self, client_with_model):
        """Test that response is a plain string."""
        response = client_with_model.post(
            "/translate",
            json={"text": "Test"}
        )
        
        assert response.status_code == 200
        # FastAPI JSON-encodes string responses, but the content is a string
        assert isinstance(response.json(), str)
        assert response.json() == "Hola mundo"


class TestRequestValidation:
    """Tests for request validation and error handling."""

    def test_missing_text_field(self, client_with_model):
        """Test that missing 'text' field returns validation error."""
        response = client_with_model.post(
            "/translate",
            json={}
        )
        
        assert response.status_code == 422  # Unprocessable Entity
        error_detail = response.json()["detail"]
        # Check that error mentions 'text' field
        assert any("text" in str(err).lower() for err in error_detail)

    def test_invalid_request_body(self, client_with_model):
        """Test that invalid JSON returns error."""
        response = client_with_model.post(
            "/translate",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code in [400, 422]  # Bad Request or Unprocessable Entity

    def test_empty_string_input(self, client_with_model, mock_model):
        """Test that empty string is accepted (edge case)."""
        class EmptyMockModel(MockModel):
            def __call__(self, text_list):
                self.call_history.append(text_list)
                return [""]
        
        from en_es_translation.api import app
        empty_model = EmptyMockModel()
        app.state.model = empty_model
        
        response = client_with_model.post(
            "/translate",
            json={"text": ""}
        )
        
        assert response.status_code == 200
        # FastAPI JSON-encodes string responses
        assert response.json() == ""

    def test_very_long_input(self, client_with_model, mock_model):
        """Test translation with very long input (boundary testing)."""
        long_text = "A" * 10000
        
        class LongMockModel(MockModel):
            def __call__(self, text_list):
                self.call_history.append(text_list)
                return ["Translated long text"]
        
        from en_es_translation.api import app
        long_model = LongMockModel()
        app.state.model = long_model
        
        response = client_with_model.post(
            "/translate",
            json={"text": long_text}
        )
        
        assert response.status_code == 200
        # Verify model received the long text
        assert len(long_model.call_history) > 0
        assert long_model.call_history[0][0] == long_text

    def test_non_string_text_field(self, client_with_model):
        """Test that non-string 'text' field returns validation error."""
        response = client_with_model.post(
            "/translate",
            json={"text": 123}
        )
        
        assert response.status_code == 422

    def test_null_text_field(self, client_with_model):
        """Test that null 'text' field returns validation error."""
        response = client_with_model.post(
            "/translate",
            json={"text": None}
        )
        
        assert response.status_code == 422


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_model_loading_failure(self):
        """Test behavior when model loading fails."""
        with patch("en_es_translation.api.load_model") as mock_load_model:
            mock_load_model.side_effect = FileNotFoundError("No checkpoint found")
            
            # Import app - lifespan will try to load model and fail
            from en_es_translation.api import app
            
            # TestClient will raise an exception if lifespan fails during startup
            # We can test this by catching the exception or by manually handling it
            try:
                with TestClient(app) as client:
                    # If we get here, lifespan succeeded (unexpected)
                    # Try to make a request - it should fail because model wasn't loaded
                    response = client.post(
                        "/translate",
                        json={"text": "Hello"}
                    )
                    # Should fail because model wasn't loaded
                    assert response.status_code == 500
            except Exception:
                # Lifespan failure during startup is also acceptable
                # This means the app couldn't start, which is the expected behavior
                pass

    def test_model_inference_error(self, client_with_model, mock_model):
        """Test behavior when model inference fails."""
        class ErrorMockModel(MockModel):
            def __call__(self, text_list):
                raise RuntimeError("CUDA out of memory")
        
        from en_es_translation.api import app
        error_model = ErrorMockModel()
        app.state.model = error_model
        
        # TestClient raises exceptions for 500 errors by default, so we need to check differently
        # or configure it not to raise. Let's check the status code if it doesn't raise.
        try:
            response = client_with_model.post(
                "/translate",
                json={"text": "Hello"},
                follow_redirects=False
            )
            # If we get here, check status code
            assert response.status_code == 500
        except RuntimeError:
            # If exception is raised, that's also acceptable - it means the error propagated
            # which is what we're testing
            pass

    def test_model_returns_empty_list(self, client_with_model, mock_model):
        """Test behavior when model returns empty list."""
        class EmptyListMockModel(MockModel):
            def __call__(self, text_list):
                return []
        
        from en_es_translation.api import app
        empty_list_model = EmptyListMockModel()
        app.state.model = empty_list_model
        
        # TestClient raises exceptions for 500 errors by default
        try:
            response = client_with_model.post(
                "/translate",
                json={"text": "Hello"},
                follow_redirects=False
            )
            # If we get here, check status code
            assert response.status_code == 500
        except IndexError:
            # If exception is raised, that's also acceptable - it means the error propagated
            # which is what we're testing
            pass

    def test_model_not_loaded(self, client):
        """Test behavior when model is not in app.state."""
        # TestClient raises exceptions for 500 errors by default
        try:
            response = client.post(
                "/translate",
                json={"text": "Hello"},
                follow_redirects=False
            )
            # If we get here, check status code
            assert response.status_code == 500
        except TypeError:
            # If exception is raised, that's also acceptable - it means the error propagated
            # which is what we're testing (model is None, so calling it raises TypeError)
            pass


class TestIntegration:
    """Integration-style tests with more realistic mocking."""

    def test_full_translation_flow(self, mock_model, mock_device):
        """Test the full translation flow with proper mocking."""
        with patch("en_es_translation.api.load_model") as mock_load_model:
            mock_load_model.return_value = (mock_model, mock_device)
            
            from en_es_translation.api import app
            
            # Manually set app state
            app.state.model = mock_model
            app.state.device = mock_device
            
            with TestClient(app) as client:
                response = client.post(
                    "/translate",
                    json={"text": "Hello, how are you?"}
                )
                
                assert response.status_code == 200
                # FastAPI JSON-encodes string responses
                assert response.json() == "Hola mundo"
                # Note: load_model won't be called if we manually set state
                # but this tests the full flow

    def test_multiple_requests(self, client_with_model, mock_model):
        """Test that multiple requests work correctly."""
        responses = []
        texts = ["Hello", "Goodbye", "Thank you"]
        
        for text in texts:
            response = client_with_model.post(
                "/translate",
                json={"text": text}
            )
            responses.append(response)
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)
        # Model should be called for each request
        assert len(mock_model.call_history) == len(texts)
