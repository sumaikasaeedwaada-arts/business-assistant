from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CustomerInsightsRequest(BaseModel):
    date_from: date | None = None
    date_to: date | None = None
    customer_identifier: str | None = Field(default=None, max_length=200)

    @field_validator("customer_identifier")
    @classmethod
    def normalize_customer_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def validate_date_range(self):
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be on or before date_to.")
        return self


class CustomerInsight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1, max_length=100)
    occurrence_count: int = Field(ge=2)
    distinct_document_count: int = Field(ge=2)
    first_record_upload: str | None = None
    last_record_upload: str | None = None


class ServiceOpportunity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    opportunity: str = Field(min_length=1, max_length=500)
    evidence_topic: str = Field(min_length=1, max_length=100)
    supporting_document_count: int = Field(ge=2)
    basis: Literal["AI interpretation"] = "AI interpretation"


class CustomerInsights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frequently_reported_issues: list[CustomerInsight] = Field(max_length=20)
    interaction_patterns: list[CustomerInsight] = Field(max_length=20)
    feedback_themes: list[CustomerInsight] = Field(max_length=20)
    transaction_patterns: list[CustomerInsight] = Field(max_length=20)
    service_opportunities: list[ServiceOpportunity] = Field(max_length=20)
    data_limitations: list[str] = Field(max_length=20)


class CustomerInsightsResponse(BaseModel):
    success: Literal[True] = True
    insights: CustomerInsights
