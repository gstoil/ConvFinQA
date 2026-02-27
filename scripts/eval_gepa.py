import asyncio
from dataclasses import dataclass
from pathlib import Path

import dotenv
from pydantic_ai import Agent
from pydantic_ai_optimizers import Optimizer
from pydantic_ai_optimizers.agents.reflection_agent import make_reflection_agent
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic import BaseModel, Field


dotenv.load_dotenv()


class Product(BaseModel):
    product_name: str = Field(description='Name of the product')
    price: float = Field(description='Numeric price')
    currency: str = Field(description='Currency code like USD or EUR')


@dataclass
class TaskInput:
    """Input to the contact extraction task."""

    text: str


@dataclass
class FieldAccuracyEvaluator(Evaluator[TaskInput, Product]):
    def evaluate(self, ctx: EvaluatorContext) -> float:
        score = 0
        expected = ctx.expected_output
        output = ctx.output

        if output.product_name == expected.product_name:
            score += 1

        if abs(output.price - expected.price) < 0.01:
            score += 1

        if output.currency == expected.currency:
            score += 1

        return score / 3.0


dataset = Dataset(
    cases=[
        Case(
            inputs='The new iPhone 15 costs $999.',
            expected_output=Product(product_name='iPhone 15', price=999.0, currency='USD'),
        ),
        Case(
            inputs='Samsung TV available now 30% less that the initial price of 799 EUR.',
            expected_output=Product(product_name='Samsung TV', price=559.3, currency='EUR'),
        ),
    ],
    evaluators=[FieldAccuracyEvaluator()],
)

reflection_agent = make_reflection_agent(model='openai:gpt-4.1-mini')

agent = Agent(
    model='openai:gpt-4.1-mini',
    output_type=Product,
    system_prompt='Extract contact information from the provided text.',
)


def run_task(system_prompt: str, text: str) -> Product:
    agent_in = Agent(
        model='openai:gpt-4o-mini',
        output_type=Product,
        system_prompt=system_prompt,
    )
    result = agent_in.run_sync(text)
    return result.output


async def extract_contact_info(input_txt: TaskInput) -> Product:
    """Run the contact extraction agent on the input text."""
    result = await agent.run(input_txt.text)
    return result.output


async def main():

    optimizer = Optimizer(
        run_case=run_task,
        dataset=dataset,
        reflection_agent=reflection_agent,
    )

    result = await optimizer.optimize(
        seed_prompt_file=Path('./data/seed_prompt.txt'), full_validation_budget=2  # small for demo
    )

    print('\nBest Prompt Found:\n')
    print(result)


asyncio.run(main())
