"""Tests for service singletons in api/services.py."""

import pytest
import threading
from unittest.mock import patch, MagicMock


class TestPersonasSingleton:
    """Tests for persona loading and caching."""

    def test_get_personas_returns_dict(self, reset_singletons):
        """get_personas returns a dict keyed by persona id."""
        from api.services import get_personas

        personas = get_personas()

        assert isinstance(personas, dict)
        assert "tech_pm" in personas
        assert "tech_swe" in personas

    def test_get_personas_list_returns_list(self, reset_singletons):
        """get_personas_list returns a list of persona dicts."""
        from api.services import get_personas_list

        personas = get_personas_list()

        assert isinstance(personas, list)
        assert len(personas) > 0
        assert all("id" in p for p in personas)

    def test_personas_cached_on_second_call(self, reset_singletons):
        """Personas are cached and same object returned on second call."""
        from api.services import get_personas

        personas1 = get_personas()
        personas2 = get_personas()

        assert personas1 is personas2

    def test_personas_thread_safe(self, reset_singletons):
        """Multiple threads get the same personas instance."""
        from api.services import get_personas

        results = []
        errors = []

        def get_and_store():
            try:
                p = get_personas()
                results.append(id(p))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_and_store) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(set(results)) == 1  # All threads got same object


class TestServiceSingletons:
    """Tests for LLM service singletons."""

    def test_get_generator_returns_singleton(self, reset_singletons):
        """get_generator returns the same instance on multiple calls."""
        import api.services as services
        mock_llm = MagicMock()
        mock_gen = MagicMock()

        with patch.object(services, "LLMClient", return_value=mock_llm):
            with patch.object(services, "ChipGenerator", return_value=mock_gen):
                gen1 = services.get_generator()
                gen2 = services.get_generator()

                assert gen1 is gen2
                assert gen1 is mock_gen

    def test_get_selector_returns_singleton(self, reset_singletons):
        """get_selector returns the same instance on multiple calls."""
        import api.services as services
        mock_llm = MagicMock()
        mock_sel = MagicMock()

        with patch.object(services, "LLMClient", return_value=mock_llm):
            with patch.object(services, "ChipSelector", return_value=mock_sel):
                sel1 = services.get_selector()
                sel2 = services.get_selector()

                assert sel1 is sel2
                assert sel1 is mock_sel

    def test_get_fill_service_returns_singleton(self, reset_singletons):
        """get_fill_service returns the same instance on multiple calls."""
        import api.services as services
        mock_llm = MagicMock()
        mock_fill = MagicMock()

        with patch.object(services, "LLMClient", return_value=mock_llm):
            with patch.object(services, "FillService", return_value=mock_fill):
                fill1 = services.get_fill_service()
                fill2 = services.get_fill_service()

                assert fill1 is fill2
                assert fill1 is mock_fill

    def test_get_llm_client_returns_singleton(self, reset_singletons):
        """get_llm_client returns the same instance on multiple calls."""
        import api.services as services
        mock_llm = MagicMock()

        with patch.object(services, "LLMClient", return_value=mock_llm):
            client1 = services.get_llm_client()
            client2 = services.get_llm_client()

            assert client1 is client2
            assert client1 is mock_llm

    def test_services_share_llm_client(self, reset_singletons):
        """All services share the same LLM client instance."""
        import api.services as services
        mock_llm = MagicMock()
        call_count = {"count": 0}

        def counting_llm():
            call_count["count"] += 1
            return mock_llm

        with patch.object(services, "LLMClient", side_effect=counting_llm):
            with patch.object(services, "ChipGenerator", return_value=MagicMock()):
                with patch.object(services, "ChipSelector", return_value=MagicMock()):
                    with patch.object(services, "FillService", return_value=MagicMock()):
                        services.get_generator()
                        services.get_selector()
                        services.get_fill_service()

                        # LLMClient should only be created once
                        assert call_count["count"] == 1


class TestThreadSafety:
    """Tests for thread safety of singletons."""

    def test_concurrent_access_same_instance(self, reset_singletons):
        """Concurrent access returns same singleton instance."""
        import api.services as services
        mock_llm = MagicMock()
        call_count = {"count": 0}

        def counting_llm():
            call_count["count"] += 1
            return mock_llm

        with patch.object(services, "LLMClient", side_effect=counting_llm):
            results = []
            errors = []

            def get_client():
                try:
                    c = services.get_llm_client()
                    results.append(id(c))
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=get_client) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(set(results)) == 1  # All threads got same object
            assert call_count["count"] == 1  # Only created once

    def test_no_double_initialization(self, reset_singletons):
        """Double-check locking prevents double initialization."""
        import api.services as services
        init_count = {"count": 0}

        def counting_init():
            init_count["count"] += 1
            return MagicMock()

        with patch.object(services, "LLMClient", side_effect=counting_init):
            threads = [threading.Thread(target=services.get_llm_client) for _ in range(50)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert init_count["count"] == 1
