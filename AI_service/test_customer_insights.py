import json
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from AI_service import api
from AI_service.dependencies import authenticated_user
from AI_service.services import customer_insights_service


class CustomerInsightsApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            "DATABASE_PATH": os.path.join(
                self.temp_directory.name,
                "customer-insights.sqlite"
            ),
            "AUTH_SECRET": "test-auth-secret-not-for-production"
        })
        self.env.start()
        api.database.initialize_database()
        self.client = TestClient(api.app)
        self.client.__enter__()
        registration = self.client.post("/api/auth/register", json={
            "email": "insights-owner@example.com",
            "password": "a-secure-test-password"
        })
        self.assertEqual(registration.status_code, 201)
        self.user_id = registration.json()["user"]["id"]
        self.headers = {
            "Authorization": f"Bearer {registration.json()['access_token']}"
        }

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.env.stop()
        self.temp_directory.cleanup()

    def add_record(
        self,
        document_id: str,
        content: str,
        uploaded_at: str,
        user_id: int | None = None
    ):
        if user_id is None:
            user_id = self.user_id
        with api.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO documents
                    (id, user_id, filename, media_type, status, created_at)
                VALUES (?, ?, ?, 'text/plain', 'ready', ?)
                """,
                (document_id, user_id, f"{document_id}.txt", uploaded_at)
            )
            connection.execute(
                """
                INSERT INTO chunks
                    (document_id, chunk_index, page_number, content, embedding_json)
                VALUES (?, 0, 1, ?, '[]')
                """,
                (document_id, content)
            )

    def test_patterns_are_measured_across_distinct_owned_records(self):
        self.add_record(
            "customer-one",
            (
                "Complaint: late delivery. Late delivery again. I called support "
                "after a slow response. Customer feedback: dissatisfied. "
                "Purchase and payment invoice. contact: private@example.com"
            ),
            "2026-10-20T12:00:00+00:00"
        )
        self.add_record(
            "customer-two",
            (
                "Complaint: late delivery. I contacted support after a slow "
                "response. Customer feedback: dissatisfied. Order was paid "
                "by invoice. C-104"
            ),
            "2026-10-25T12:00:00+00:00"
        )
        other_user = self.client.post("/api/auth/register", json={
            "email": "different-owner@example.com",
            "password": "a-secure-test-password"
        }).json()["user"]["id"]
        self.add_record(
            "other-owner",
            "Complaint: late delivery.",
            "2026-10-25T12:00:00+00:00",
            user_id=other_user
        )
        model_output = json.dumps([
            {
                "evidence_topic": "Delivery delays",
                "opportunity": (
                    "Review delivery handling in light of the repeated records."
                )
            }
        ])
        with patch.object(
            api.ai_service,
            "generate_answer",
            return_value=model_output
        ) as generate:
            response = self.client.post(
                "/api/customer-insights",
                headers=self.headers,
                json={}
            )

        self.assertEqual(response.status_code, 200, response.text)
        insights = response.json()["insights"]
        delivery = next(
            item for item in insights["frequently_reported_issues"]
            if item["topic"] == "Delivery delays"
        )
        self.assertEqual(delivery["occurrence_count"], 3)
        self.assertEqual(delivery["distinct_document_count"], 2)
        self.assertEqual(delivery["first_record_upload"], "2026-10-20T12:00:00+00:00")
        self.assertEqual(delivery["last_record_upload"], "2026-10-25T12:00:00+00:00")
        self.assertIn(
            "Negative feedback",
            [item["topic"] for item in insights["feedback_themes"]]
        )
        self.assertIn(
            "Support contacts",
            [item["topic"] for item in insights["interaction_patterns"]]
        )
        self.assertIn(
            "Payments or invoices",
            [item["topic"] for item in insights["transaction_patterns"]]
        )
        self.assertEqual(
            insights["service_opportunities"][0]["supporting_document_count"],
            2
        )
        prompt = generate.call_args.args[0]
        self.assertNotIn("private@example.com", prompt)
        self.assertNotIn("C-104", prompt)
        self.assertNotIn("late delivery again", prompt.lower())
        self.assertIn("distinct_uploaded_document_count", prompt)
        self.assertTrue(any(
            "no normalized customer" in item.lower()
            for item in insights["data_limitations"]
        ))

    def test_single_record_is_not_reported_as_a_recurring_pattern(self):
        self.add_record(
            "single",
            "One customer reported a late delivery and requested a refund.",
            "2026-10-20T12:00:00+00:00"
        )
        with patch.object(api.ai_service, "generate_answer") as generate:
            response = self.client.post(
                "/api/customer-insights",
                headers=self.headers,
                json={}
            )

        self.assertEqual(response.status_code, 200)
        insights = response.json()["insights"]
        self.assertEqual(insights["frequently_reported_issues"], [])
        self.assertEqual(insights["feedback_themes"], [])
        self.assertEqual(insights["service_opportunities"], [])
        self.assertTrue(any(
            "no recurring issue" in item.lower()
            for item in insights["data_limitations"]
        ))
        generate.assert_not_called()

    def test_overlapping_chunks_do_not_inflate_occurrence_counts(self):
        self.add_record(
            "overlapping",
            "Complaint: late delivery",
            "2026-10-20T12:00:00+00:00"
        )
        with api.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO chunks
                    (document_id, chunk_index, page_number, content, embedding_json)
                VALUES ('overlapping', 1, 1, 'late delivery', '[]')
                """
            )
        self.add_record(
            "separate",
            "Complaint: late delivery.",
            "2026-10-21T12:00:00+00:00"
        )
        with patch.object(
            api.ai_service,
            "generate_answer",
            return_value=json.dumps([{
                "evidence_topic": "Delivery delays",
                "opportunity": "Review repeated delivery delay records."
            }])
        ):
            response = self.client.post(
                "/api/customer-insights",
                headers=self.headers,
                json={}
            )

        self.assertEqual(response.status_code, 200, response.text)
        delivery = response.json()["insights"]["frequently_reported_issues"][0]
        self.assertEqual(delivery["occurrence_count"], 2)
        self.assertEqual(delivery["distinct_document_count"], 2)

    def test_empty_dataset_returns_limitations_without_calling_ai(self):
        with patch.object(api.ai_service, "generate_answer") as generate:
            response = self.client.post(
                "/api/customer-insights",
                headers=self.headers,
                json={}
            )

        self.assertEqual(response.status_code, 200)
        insights = response.json()["insights"]
        self.assertEqual(insights["frequently_reported_issues"], [])
        self.assertTrue(any(
            "no matching uploaded document" in item.lower()
            for item in insights["data_limitations"]
        ))
        generate.assert_not_called()

    def test_customer_and_upload_date_filters_are_applied(self):
        self.add_record(
            "matching-old",
            "C-104 reported a late delivery.",
            "2026-10-20T12:00:00+00:00"
        )
        self.add_record(
            "matching-current",
            "C-104 reported a late delivery.",
            "2026-10-25T12:00:00+00:00"
        )
        self.add_record(
            "other-customer",
            "C-105 reported a late delivery.",
            "2026-10-25T12:00:00+00:00"
        )
        with patch.object(
            api.ai_service,
            "generate_answer",
            return_value=json.dumps([{
                "evidence_topic": "Delivery delays",
                "opportunity": "Review repeated delivery delay records."
            }])
        ) as generate:
            response = self.client.post(
                "/api/customer-insights",
                headers=self.headers,
                json={
                    "customer_identifier": " C-104 ",
                    "date_from": "2026-10-20",
                    "date_to": "2026-10-25"
                }
            )

        self.assertEqual(response.status_code, 200, response.text)
        insights = response.json()["insights"]
        delivery = insights["frequently_reported_issues"][0]
        self.assertEqual(delivery["distinct_document_count"], 2)
        prompt = generate.call_args.args[0]
        self.assertIn("Delivery delays", prompt)

    def test_invalid_date_range_and_missing_authentication_are_rejected(self):
        invalid = self.client.post(
            "/api/customer-insights",
            headers=self.headers,
            json={"date_from": "2026-10-21", "date_to": "2026-10-20"}
        )
        unauthenticated = self.client.post("/api/customer-insights", json={})

        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(unauthenticated.status_code, 401)

    def test_unavailable_ai_is_reported_without_provider_details(self):
        self.add_record(
            "one",
            "Complaint: late delivery.",
            "2026-10-20T12:00:00+00:00"
        )
        self.add_record(
            "two",
            "Complaint: late delivery.",
            "2026-10-21T12:00:00+00:00"
        )
        with patch.object(
            api.ai_service,
            "generate_answer",
            side_effect=TimeoutError("private provider credential")
        ):
            response = self.client.post(
                "/api/customer-insights",
                headers=self.headers,
                json={}
            )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("credential", response.text)

    def test_invalid_ai_opportunity_is_rejected(self):
        self.add_record(
            "one",
            "Complaint: late delivery.",
            "2026-10-20T12:00:00+00:00"
        )
        self.add_record(
            "two",
            "Complaint: late delivery.",
            "2026-10-21T12:00:00+00:00"
        )
        with patch.object(
            api.ai_service,
            "generate_answer",
            return_value=json.dumps([{
                "evidence_topic": "Invented customer preference",
                "opportunity": "Offer an unsupported service."
            }])
        ):
            response = self.client.post(
                "/api/customer-insights",
                headers=self.headers,
                json={}
            )

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("Invented customer preference", response.text)

    def test_database_failure_returns_service_unavailable(self):
        with patch.object(
            customer_insights_service,
            "_retrieve_records",
            side_effect=customer_insights_service.CustomerInsightsUnavailableError()
        ):
            response = self.client.post(
                "/api/customer-insights",
                headers=self.headers,
                json={}
            )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private database path", response.text)


if __name__ == "__main__":
    unittest.main()
