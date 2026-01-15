import pytest
import sqlite3
import json
from pathlib import Path

from db.repository import Repository, Run


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    schema_path = Path(__file__).parent.parent / "db" / "schema.sql"
    conn.executescript(schema_path.read_text())
    conn.commit()
    return conn


@pytest.fixture
def repo(db):
    return Repository(db=db)


class TestRunOperations:
    def test_create_run(self, repo):
        run_id = repo.create_run()
        assert run_id is not None
        assert len(run_id) == 36

    def test_create_run_with_config(self, repo):
        config = {"models": ["model1", "model2"], "styles": ["terse"]}
        run_id = repo.create_run(config=config)

        run = repo.get_run(run_id)
        assert run is not None
        assert run.config == config

    def test_get_run(self, repo):
        run_id = repo.create_run()
        run = repo.get_run(run_id)

        assert run is not None
        assert run.id == run_id
        assert run.status == "running"
        assert run.completed_at is None

    def test_get_run_not_found(self, repo):
        run = repo.get_run("nonexistent-id")
        assert run is None

    def test_complete_run(self, repo):
        run_id = repo.create_run()
        repo.complete_run(run_id, status="completed")

        run = repo.get_run(run_id)
        assert run.status == "completed"
        assert run.completed_at is not None

    def test_complete_run_failed(self, repo):
        run_id = repo.create_run()
        repo.complete_run(run_id, status="failed")

        run = repo.get_run(run_id)
        assert run.status == "failed"

    def test_list_runs(self, repo):
        run_id1 = repo.create_run()
        run_id2 = repo.create_run()

        runs, total = repo.list_runs()
        assert total == 2
        assert len(runs) == 2

    def test_list_runs_with_pagination(self, repo):
        for _ in range(5):
            repo.create_run()

        runs, total = repo.list_runs(limit=2, offset=0)
        assert total == 5
        assert len(runs) == 2

        runs2, _ = repo.list_runs(limit=2, offset=2)
        assert len(runs2) == 2


class TestResultOperations:
    def test_save_result(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        result_id = repo.save_result(
            run_id=run_id,
            model="test-model",
            persona_id="p1",
            sector="Tech",
            desired_role="Dev",
            style="terse",
            input_type="basic",
            constraint_type="no_constraint",
            chip_count=15,
            final_chips=chips,
        )

        assert result_id is not None
        assert len(result_id) == 36

    def test_save_result_with_all_fields(self, repo):
        run_id = repo.create_run()
        final_chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]
        step1_chips = [{"key": "s1", "display": "Step 1", "type": "jargon"}]
        selected_chips = [{"key": "sel1", "display": "Selected", "type": "role_task"}]
        fill_chips = [{"key": "f1", "display": "Fill", "type": "environment"}]

        result_id = repo.save_result(
            run_id=run_id,
            model="test-model",
            persona_id="p1",
            sector="Tech",
            desired_role="Dev",
            style="guided",
            input_type="enriched",
            constraint_type="with_constraint",
            chip_count=35,
            final_chips=final_chips,
            step1_chips=step1_chips,
            selected_chips=selected_chips,
            fill_chips=fill_chips,
            errors=["error1", "error2"],
            latency_ms=500,
            input_tokens=1000,
            output_tokens=200,
            cost_usd=0.0015,
        )

        result = repo.get_result(result_id)
        assert result is not None
        assert result["step1_chips"] == step1_chips
        assert result["selected_chips"] == selected_chips
        assert result["fill_chips"] == fill_chips
        assert result["errors"] == ["error1", "error2"]
        assert result["latency_ms"] == 500
        assert result["cost_usd"] == 0.0015

    def test_result_exists(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        repo.save_result(
            run_id=run_id,
            model="test-model",
            persona_id="p1",
            sector="Tech",
            desired_role="Dev",
            style="terse",
            input_type="basic",
            constraint_type="no_constraint",
            chip_count=15,
            final_chips=chips,
        )

        assert repo.result_exists(
            run_id=run_id,
            model="test-model",
            persona_id="p1",
            style="terse",
            input_type="basic",
            constraint_type="no_constraint",
            chip_count=15,
        )

    def test_result_not_exists(self, repo):
        run_id = repo.create_run()

        assert not repo.result_exists(
            run_id=run_id,
            model="other-model",
            persona_id="p1",
            style="terse",
            input_type="basic",
            constraint_type="no_constraint",
            chip_count=15,
        )

    def test_get_result(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        result_id = repo.save_result(
            run_id=run_id,
            model="test-model",
            persona_id="p1",
            sector="Tech",
            desired_role="Dev",
            style="terse",
            input_type="basic",
            constraint_type="no_constraint",
            chip_count=15,
            final_chips=chips,
            latency_ms=100,
        )

        result = repo.get_result(result_id)
        assert result is not None
        assert result["model"] == "test-model"
        assert result["final_chips"] == chips
        assert result["latency_ms"] == 100

    def test_get_result_not_found(self, repo):
        result = repo.get_result("nonexistent-id")
        assert result is None

    def test_list_results_no_filters(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        for i in range(3):
            repo.save_result(
                run_id=run_id,
                model=f"model-{i}",
                persona_id="p1",
                sector="Tech",
                desired_role="Dev",
                style="terse",
                input_type="basic",
                constraint_type="no_constraint",
                chip_count=15,
                final_chips=chips,
            )

        results, total = repo.list_results()
        assert total == 3
        assert len(results) == 3

    def test_list_results_filter_by_model(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        repo.save_result(run_id=run_id, model="model-a", persona_id="p1", sector="Tech",
                         desired_role="Dev", style="terse", input_type="basic",
                         constraint_type="no_constraint", chip_count=15, final_chips=chips)
        repo.save_result(run_id=run_id, model="model-b", persona_id="p1", sector="Tech",
                         desired_role="Dev", style="guided", input_type="basic",
                         constraint_type="no_constraint", chip_count=15, final_chips=chips)

        results, total = repo.list_results(model="model-a")
        assert total == 1
        assert results[0]["model"] == "model-a"

    def test_list_results_filter_by_style(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        repo.save_result(run_id=run_id, model="model-a", persona_id="p1", sector="Tech",
                         desired_role="Dev", style="terse", input_type="basic",
                         constraint_type="no_constraint", chip_count=15, final_chips=chips)
        repo.save_result(run_id=run_id, model="model-a", persona_id="p1", sector="Tech",
                         desired_role="Dev", style="guided", input_type="enriched",
                         constraint_type="no_constraint", chip_count=35, final_chips=chips)

        results, total = repo.list_results(style="guided")
        assert total == 1
        assert results[0]["style"] == "guided"

    def test_list_results_filter_by_chip_count(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        repo.save_result(run_id=run_id, model="model-a", persona_id="p1", sector="Tech",
                         desired_role="Dev", style="terse", input_type="basic",
                         constraint_type="no_constraint", chip_count=15, final_chips=chips)
        repo.save_result(run_id=run_id, model="model-a", persona_id="p1", sector="Tech",
                         desired_role="Dev", style="terse", input_type="enriched",
                         constraint_type="no_constraint", chip_count=35, final_chips=chips)

        results, total = repo.list_results(chip_count=35)
        assert total == 1
        assert results[0]["chip_count"] == 35

    def test_list_results_multiple_filters(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        repo.save_result(run_id=run_id, model="model-a", persona_id="p1", sector="Tech",
                         desired_role="Dev", style="terse", input_type="basic",
                         constraint_type="no_constraint", chip_count=15, final_chips=chips)
        repo.save_result(run_id=run_id, model="model-a", persona_id="p2", sector="Finance",
                         desired_role="Analyst", style="terse", input_type="basic",
                         constraint_type="with_constraint", chip_count=15, final_chips=chips)
        repo.save_result(run_id=run_id, model="model-b", persona_id="p1", sector="Tech",
                         desired_role="Dev", style="guided", input_type="basic",
                         constraint_type="no_constraint", chip_count=15, final_chips=chips)

        results, total = repo.list_results(model="model-a", style="terse")
        assert total == 2

    def test_list_results_pagination(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        for i in range(10):
            repo.save_result(run_id=run_id, model=f"model-{i}", persona_id="p1", sector="Tech",
                             desired_role="Dev", style="terse", input_type=f"type-{i}",
                             constraint_type="no_constraint", chip_count=15, final_chips=chips)

        results, total = repo.list_results(limit=3, offset=0)
        assert total == 10
        assert len(results) == 3


class TestRatingOperations:
    def test_add_rating(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        result_id = repo.save_result(run_id=run_id, model="model-a", persona_id="p1",
                                     sector="Tech", desired_role="Dev", style="terse",
                                     input_type="basic", constraint_type="no_constraint",
                                     chip_count=15, final_chips=chips)

        rating_id, created_at = repo.add_rating(result_id, "user1", 5)
        assert rating_id is not None
        assert created_at is not None

    def test_add_rating_upsert(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        result_id = repo.save_result(run_id=run_id, model="model-a", persona_id="p1",
                                     sector="Tech", desired_role="Dev", style="terse",
                                     input_type="basic", constraint_type="no_constraint",
                                     chip_count=15, final_chips=chips)

        repo.add_rating(result_id, "user1", 3)
        repo.add_rating(result_id, "user1", 5)

        ratings = repo.get_ratings(result_id=result_id)
        assert len(ratings) == 1
        assert ratings[0]["rating"] == 5

    def test_get_ratings_by_result(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        result_id = repo.save_result(run_id=run_id, model="model-a", persona_id="p1",
                                     sector="Tech", desired_role="Dev", style="terse",
                                     input_type="basic", constraint_type="no_constraint",
                                     chip_count=15, final_chips=chips)

        repo.add_rating(result_id, "user1", 4)
        repo.add_rating(result_id, "user2", 5)

        ratings = repo.get_ratings(result_id=result_id)
        assert len(ratings) == 2

    def test_get_ratings_by_user(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        result1 = repo.save_result(run_id=run_id, model="model-a", persona_id="p1",
                                   sector="Tech", desired_role="Dev", style="terse",
                                   input_type="basic", constraint_type="no_constraint",
                                   chip_count=15, final_chips=chips)
        result2 = repo.save_result(run_id=run_id, model="model-a", persona_id="p2",
                                   sector="Finance", desired_role="Analyst", style="guided",
                                   input_type="basic", constraint_type="no_constraint",
                                   chip_count=15, final_chips=chips)

        repo.add_rating(result1, "user1", 4)
        repo.add_rating(result2, "user1", 5)
        repo.add_rating(result1, "user2", 3)

        ratings = repo.get_ratings(user_id="user1")
        assert len(ratings) == 2

    def test_list_results_rated_by_filter(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        result1 = repo.save_result(run_id=run_id, model="model-a", persona_id="p1",
                                   sector="Tech", desired_role="Dev", style="terse",
                                   input_type="basic", constraint_type="no_constraint",
                                   chip_count=15, final_chips=chips)
        repo.save_result(run_id=run_id, model="model-b", persona_id="p1",
                         sector="Tech", desired_role="Dev", style="guided",
                         input_type="basic", constraint_type="no_constraint",
                         chip_count=15, final_chips=chips)

        repo.add_rating(result1, "user1", 4)

        results, total = repo.list_results(rated_by="user1")
        assert total == 1
        assert results[0]["id"] == result1

    def test_list_results_unrated_by_filter(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        result1 = repo.save_result(run_id=run_id, model="model-a", persona_id="p1",
                                   sector="Tech", desired_role="Dev", style="terse",
                                   input_type="basic", constraint_type="no_constraint",
                                   chip_count=15, final_chips=chips)
        result2 = repo.save_result(run_id=run_id, model="model-b", persona_id="p1",
                                   sector="Tech", desired_role="Dev", style="guided",
                                   input_type="basic", constraint_type="no_constraint",
                                   chip_count=15, final_chips=chips)

        repo.add_rating(result1, "user1", 4)

        results, total = repo.list_results(unrated_by="user1")
        assert total == 1
        assert results[0]["id"] == result2

    def test_result_avg_rating(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        result_id = repo.save_result(run_id=run_id, model="model-a", persona_id="p1",
                                     sector="Tech", desired_role="Dev", style="terse",
                                     input_type="basic", constraint_type="no_constraint",
                                     chip_count=15, final_chips=chips)

        repo.add_rating(result_id, "user1", 4)
        repo.add_rating(result_id, "user2", 5)
        repo.add_rating(result_id, "user3", 3)

        result = repo.get_result(result_id)
        assert result["avg_rating"] == 4.0
        assert result["rating_count"] == 3


class TestStatsOperations:
    def test_get_stats_by_model(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        repo.save_result(run_id=run_id, model="model-a", persona_id="p1",
                         sector="Tech", desired_role="Dev", style="terse",
                         input_type="basic", constraint_type="no_constraint",
                         chip_count=15, final_chips=chips, cost_usd=0.001, latency_ms=100)
        repo.save_result(run_id=run_id, model="model-a", persona_id="p2",
                         sector="Finance", desired_role="Analyst", style="guided",
                         input_type="basic", constraint_type="no_constraint",
                         chip_count=15, final_chips=chips, cost_usd=0.002, latency_ms=200)
        repo.save_result(run_id=run_id, model="model-b", persona_id="p1",
                         sector="Tech", desired_role="Dev", style="terse",
                         input_type="enriched", constraint_type="no_constraint",
                         chip_count=15, final_chips=chips, cost_usd=0.003, latency_ms=300)

        stats = repo.get_stats(group_by="model")
        assert len(stats) == 2

        model_a_stats = next(s for s in stats if s["model"] == "model-a")
        assert model_a_stats["result_count"] == 2
        assert model_a_stats["total_cost_usd"] == pytest.approx(0.003, abs=0.0001)
        assert model_a_stats["avg_latency_ms"] == 150

    def test_get_stats_by_style(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        repo.save_result(run_id=run_id, model="model-a", persona_id="p1",
                         sector="Tech", desired_role="Dev", style="terse",
                         input_type="basic", constraint_type="no_constraint",
                         chip_count=15, final_chips=chips)
        repo.save_result(run_id=run_id, model="model-a", persona_id="p1",
                         sector="Tech", desired_role="Dev", style="guided",
                         input_type="enriched", constraint_type="no_constraint",
                         chip_count=15, final_chips=chips)

        stats = repo.get_stats(group_by="style")
        assert len(stats) == 2
        styles = {s["style"] for s in stats}
        assert styles == {"terse", "guided"}

    def test_get_stats_with_run_filter(self, repo):
        run_id1 = repo.create_run()
        run_id2 = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        repo.save_result(run_id=run_id1, model="model-a", persona_id="p1",
                         sector="Tech", desired_role="Dev", style="terse",
                         input_type="basic", constraint_type="no_constraint",
                         chip_count=15, final_chips=chips)
        repo.save_result(run_id=run_id2, model="model-b", persona_id="p1",
                         sector="Tech", desired_role="Dev", style="terse",
                         input_type="basic", constraint_type="no_constraint",
                         chip_count=15, final_chips=chips)

        stats = repo.get_stats(group_by="model", run_id=run_id1)
        assert len(stats) == 1
        assert stats[0]["model"] == "model-a"

    def test_get_stats_invalid_group_by(self, repo):
        with pytest.raises(ValueError, match="group_by must be one of"):
            repo.get_stats(group_by="invalid")

    def test_get_stats_with_ratings(self, repo):
        run_id = repo.create_run()
        chips = [{"key": "c1", "display": "Chip 1", "type": "situation"}]

        result_id = repo.save_result(run_id=run_id, model="model-a", persona_id="p1",
                                     sector="Tech", desired_role="Dev", style="terse",
                                     input_type="basic", constraint_type="no_constraint",
                                     chip_count=15, final_chips=chips)

        repo.add_rating(result_id, "user1", 4)
        repo.add_rating(result_id, "user2", 5)

        stats = repo.get_stats(group_by="model")
        model_a_stats = stats[0]
        assert model_a_stats["rated_count"] == 2
        assert model_a_stats["avg_rating"] == 4.5
