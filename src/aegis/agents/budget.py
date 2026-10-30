"""Token and dollar budgets. Crossing either one stops the run before the next call."""

from __future__ import annotations

from aegis.errors import BudgetExceeded


class RunBudget:
    def __init__(
        self,
        *,
        max_input_tokens: int,
        max_output_tokens: int,
        max_cost_usd: float,
        max_llm_calls: int,
        price_input_per_mtok: float,
        price_output_per_mtok: float,
    ) -> None:
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.max_cost_usd = max_cost_usd
        self.max_llm_calls = max_llm_calls
        self.price_input_per_mtok = price_input_per_mtok
        self.price_output_per_mtok = price_output_per_mtok
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.cost_usd = 0.0

    def ensure_can_call(self) -> None:
        """Hard stop before a call once a cap is already met."""
        if self.calls >= self.max_llm_calls:
            raise BudgetExceeded(
                f"Hard stop: LLM call budget is {self.max_llm_calls} and {self.calls} have already run."
            )
        if self.input_tokens >= self.max_input_tokens:
            raise BudgetExceeded(
                f"Hard stop: input token budget is {self.max_input_tokens} "
                f"and {self.input_tokens} have already been used."
            )
        if self.output_tokens >= self.max_output_tokens:
            raise BudgetExceeded(
                f"Hard stop: output token budget is {self.max_output_tokens} "
                f"and {self.output_tokens} have already been used."
            )
        if self.cost_usd >= self.max_cost_usd:
            raise BudgetExceeded(
                f"Hard stop: cost budget is ${self.max_cost_usd:.4f} "
                f"and ${self.cost_usd:.4f} has already been spent."
            )

    def charge(self, input_tokens: int, output_tokens: int) -> None:
        """Record a completed call. Raise when this call crossed a cap."""
        self.calls += 1
        self.input_tokens += max(0, input_tokens)
        self.output_tokens += max(0, output_tokens)
        self.cost_usd += (max(0, input_tokens) / 1_000_000) * self.price_input_per_mtok
        self.cost_usd += (max(0, output_tokens) / 1_000_000) * self.price_output_per_mtok
        if self.input_tokens > self.max_input_tokens:
            raise BudgetExceeded(
                f"Hard stop: input tokens {self.input_tokens} exceed the budget of {self.max_input_tokens}."
            )
        if self.output_tokens > self.max_output_tokens:
            raise BudgetExceeded(
                f"Hard stop: output tokens {self.output_tokens} exceed the budget of {self.max_output_tokens}."
            )
        if self.cost_usd > self.max_cost_usd:
            raise BudgetExceeded(
                f"Hard stop: cost ${self.cost_usd:.4f} exceeds the budget of ${self.max_cost_usd:.4f}."
            )

    def snapshot(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "max_llm_calls": self.max_llm_calls,
            "max_cost_usd": self.max_cost_usd,
        }
