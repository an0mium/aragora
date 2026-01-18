"""
Debate Step for executing Aragora debates within workflows.

Provides workflow integration with Aragora's debate orchestration:
- Execute multi-agent debates as workflow steps
- Capture consensus, synthesis, and agent responses
- Support for different debate protocols and topologies
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from aragora.workflow.step import BaseStep, WorkflowContext

logger = logging.getLogger(__name__)


class DebateStep(BaseStep):
    """
    Debate step for executing Aragora debates within workflows.

    Config options:
        topic: str - Debate topic (can use {input} placeholders)
        agents: List[str] - Agent types to use (default: auto-selected)
        rounds: int - Number of debate rounds (default: 3)
        topology: str - Debate topology (default: "round_robin")
        consensus_mechanism: str - How to determine consensus (default: "majority")
        enable_critique: bool - Enable agent critiques (default: True)
        enable_synthesis: bool - Generate synthesis at end (default: True)
        timeout_seconds: float - Timeout per round (default: 120)
        memory_enabled: bool - Use memory for context (default: True)
        tenant_id: str - Tenant for multi-tenant isolation

    Topologies:
        - "round_robin": Each agent speaks in turn
        - "graph": Free-form discussion with branching
        - "adversarial": Two opposing teams
        - "hive_mind": Parallel responses
        - "dialectic": Thesis-antithesis-synthesis
        - "socratic": Question-driven exploration

    Usage:
        step = DebateStep(
            name="Contract Review Debate",
            config={
                "topic": "Review the terms of {contract_name}",
                "agents": ["legal_analyst", "risk_assessor", "compliance_officer"],
                "rounds": 3,
                "topology": "round_robin",
                "consensus_mechanism": "unanimous",
            }
        )
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the debate step."""
        config = {**self._config, **context.current_step_config}

        # Build topic from template
        topic_template = config.get("topic", "")
        topic = self._interpolate_text(topic_template, context)

        if not topic:
            logger.warning(f"Empty topic for debate step '{self.name}'")
            return {"success": False, "error": "Empty topic"}

        try:
            from aragora import Arena, Environment, DebateProtocol
            from aragora.agents import create_agent

            # Build environment
            env = Environment(task=topic)

            # Get agents
            agent_types = config.get("agents", ["claude", "gpt4"])
            agents = []
            for agent_type in agent_types:
                try:
                    agent = create_agent(agent_type)
                    agents.append(agent)
                except Exception as e:
                    logger.warning(f"Failed to create agent {agent_type}: {e}")

            if not agents:
                return {"success": False, "error": "No agents available"}

            # Build protocol
            protocol = DebateProtocol(
                rounds=config.get("rounds", 3),
                topology=config.get("topology", "round_robin"),
                consensus=config.get("consensus_mechanism", "majority"),
                enable_critique=config.get("enable_critique", True),
                enable_synthesis=config.get("enable_synthesis", True),
            )

            # Create arena
            arena = Arena(
                env=env,
                agents=agents,
                protocol=protocol,
                timeout_seconds=config.get("timeout_seconds", 120),
            )

            # Execute debate
            logger.info(f"Starting debate '{self.name}' on topic: {topic[:100]}...")
            result = await arena.run()

            # Extract key information
            output = {
                "success": True,
                "debate_id": getattr(result, "debate_id", ""),
                "topic": topic,
                "rounds_completed": getattr(result, "rounds_completed", 0),
                "consensus_reached": getattr(result, "consensus_reached", False),
                "consensus": getattr(result, "consensus", None),
                "synthesis": getattr(result, "synthesis", None),
                "agents": [
                    {
                        "name": getattr(a, "name", str(a)),
                        "model": getattr(a, "model", "unknown"),
                    }
                    for a in agents
                ],
                "execution_time_ms": getattr(result, "duration_ms", 0),
            }

            # Include agent responses if available
            if hasattr(result, "responses"):
                output["responses"] = [
                    {
                        "agent": r.get("agent", "unknown"),
                        "round": r.get("round", 0),
                        "content": r.get("content", "")[:500],  # Truncate long responses
                    }
                    for r in result.responses[:20]  # Limit to 20 responses
                ]

            logger.info(f"Debate '{self.name}' completed. Consensus: {output['consensus_reached']}")
            return output

        except ImportError as e:
            logger.warning(f"Aragora debate components not available: {e}")
            return {"success": False, "error": f"Debate components not available: {e}"}

        except Exception as e:
            logger.error(f"Debate execution failed: {e}")
            return {"success": False, "error": str(e)}

    def _interpolate_text(self, template: str, context: WorkflowContext) -> str:
        """Interpolate template with context values."""
        text = template

        # Replace {input_name} with input values
        for key, value in context.inputs.items():
            text = text.replace(f"{{{key}}}", str(value))

        # Replace {step.step_id} with step outputs
        for step_id, output in context.step_outputs.items():
            if isinstance(output, str):
                text = text.replace(f"{{step.{step_id}}}", output)
            elif isinstance(output, dict):
                for sub_key in ["response", "content", "result", "synthesis"]:
                    if sub_key in output:
                        text = text.replace(f"{{step.{step_id}}}", str(output[sub_key]))
                        break

        # Replace {state.key} with state values
        for key, value in context.state.items():
            text = text.replace(f"{{state.{key}}}", str(value))

        return text


class QuickDebateStep(BaseStep):
    """
    Lightweight debate step for quick multi-agent consultations.

    Uses a simplified debate format without full orchestration:
    - Single round of responses from each agent
    - Simple majority synthesis
    - Faster execution for time-sensitive workflows

    Config options:
        question: str - Question to ask agents
        agents: List[str] - Agent types (default: 2 agents)
        max_response_length: int - Max chars per response (default: 500)
        synthesize: bool - Generate synthesis (default: True)
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute quick debate."""
        config = {**self._config, **context.current_step_config}

        question_template = config.get("question", "")
        question = self._interpolate_text(question_template, context)

        if not question:
            return {"success": False, "error": "Empty question"}

        try:
            import asyncio
            from aragora.agents import create_agent

            agent_types = config.get("agents", ["claude", "gpt4"])
            max_length = config.get("max_response_length", 500)

            # Get responses in parallel
            async def get_response(agent_type: str) -> Dict[str, Any]:
                try:
                    agent = create_agent(agent_type)
                    response = await agent.generate(question)
                    return {
                        "agent": agent_type,
                        "response": response[:max_length] if len(response) > max_length else response,
                        "success": True,
                    }
                except Exception as e:
                    return {"agent": agent_type, "error": str(e), "success": False}

            responses = await asyncio.gather(*[get_response(at) for at in agent_types])

            # Filter successful responses
            successful = [r for r in responses if r["success"]]

            # Simple synthesis if enabled
            synthesis = None
            if config.get("synthesize", True) and len(successful) > 1:
                # Use first agent to synthesize
                try:
                    synth_agent = create_agent(agent_types[0])
                    synth_prompt = f"Synthesize these perspectives on: {question}\n\n"
                    for r in successful:
                        synth_prompt += f"- {r['agent']}: {r['response']}\n\n"
                    synth_prompt += "Provide a brief synthesis:"
                    synthesis = await synth_agent.generate(synth_prompt)
                except Exception:
                    synthesis = None

            return {
                "success": len(successful) > 0,
                "question": question,
                "responses": responses,
                "synthesis": synthesis,
                "agents_responded": len(successful),
                "agents_total": len(agent_types),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _interpolate_text(self, template: str, context: WorkflowContext) -> str:
        """Interpolate template with context values."""
        text = template
        for key, value in context.inputs.items():
            text = text.replace(f"{{{key}}}", str(value))
        for step_id, output in context.step_outputs.items():
            if isinstance(output, str):
                text = text.replace(f"{{step.{step_id}}}", output)
        return text
